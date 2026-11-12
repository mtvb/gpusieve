// gpusieve: special-q lattice siever on the GPU for one side of a CADO poly file, the shared side of a factorization factory:
// either the rational side Y1*x + Y0 (side 0, any Y0, Y1) or the algebraic side c_d*x^d + ... + c0 (side 1, degree <= 8, any
// coefficients). Only that side is sieved, with the special-q on it; the other side must be linear, and if its coefficients
// fit into 64 bits it is taken as a dummy side that is only checked to stay below 2^dummybits (default 32, as with
// lpb = mfb = 32 in las; -dummybits 0: not checked).
//
// Survivors, the coprime (a,b) whose sieved log sum comes within mfb+slack bits of log2|F(a,b)/q|, are written to an .ab
// file whose header names the other side (NFS_factory rebuilds it) for the exact check by NFS_factory with "side:" the sieved
// side.
//
// Conventions follow CADO-NFS las: the special-q lattice is reduced with the skewed Gauss reduction and J is chosen as by
// las' default adjust strategy; (a,b) = i*(a0,b0) + j*(a1,b1) with -I/2 <= i < I/2, 1 <= j < J; large primes are
// enumerated with the Franke-Kleinjung basis (warp vector (-mi0,j0), step vector (i1,j1)).
//
//   gpusieve -poly trick.poly -I 15 -lim 25000000 -mfb 56 [-slack 12] [-dummybits 32] -q0 Q0 -q1 Q1 [-random N] -out surv.ab
//   gpusieve ... -qfile lasq.txt -out surv.ab      special-q, basis and J taken from las "# Sieving side-S q=..." lines
//   -side 0|1: the side to sieve (default: the algebraic side if its degree is 2 or more, else the rational side)
//   -I 10..19; -parts N: sieve each special-q in N passes over its rows (N a power of 2; default: as few as fit into the
//   free GPU memory), with the buckets of one pass at a time
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

typedef unsigned __int128 u128;
typedef __int128 i128;
#define MAXD 8                  // largest degree of the sieved polynomial

struct Poly {
	std::string Y0s, Y1s = "1";   // rational side Y1*x + Y0, any size (decimal strings)
	std::string c[MAXD + 1];      // algebraic side c0 .. c8, any size (empty: absent)
	int alg_deg = -1;             // its degree (-1: the file has none)
	double skew = 1;
};

static bool zero_str(const std::string& s) { for (char ch : s) if (ch >= '1' && ch <= '9') return false; return true; }

static Poly read_poly(const char* path)
{
	Poly P;
	FILE* f = fopen(path, "r");
	if (!f) { perror(path); exit(1); }
	char line[4096];
	while (fgets(line, sizeof line, f)) {
		char key[64], val[4000];
		if (sscanf(line, " %63[^: ] : %3999s", key, val) != 2) continue;
		std::string k = key;
		if (k == "Y0") P.Y0s = val;
		else if (k == "Y1") P.Y1s = val;
		else if (k.size() == 2 && k[0] == 'c' && k[1] >= '0' && k[1] <= '9') {
			int d = k[1] - '0';
			if (d > MAXD) { if (!zero_str(val)) { fprintf(stderr, "%s: degree %d, at most %d is supported\n", path, d, MAXD); exit(1); } }
			else P.c[d] = val;
		}
		else if (k == "skew") P.skew = atof(val);
	}
	fclose(f);
	for (int d = MAXD; d >= 0; d--) if (!P.c[d].empty() && !zero_str(P.c[d])) { P.alg_deg = d; break; }
	return P;
}

// (decimal string) mod m, for m < 2^63
static uint64_t mod_str(const std::string& s, uint64_t m)
{
	uint64_t r = 0; bool neg = false;
	const bool small = m < (1ull << 32);
	for (char ch : s) {
		if (ch == '-') { neg = true; continue; }
		if (ch < '0' || ch > '9') continue;
		r = small ? (r * 10 + (uint64_t)(ch - '0')) % m : (uint64_t)(((u128)r * 10 + (uint64_t)(ch - '0')) % m);
	}
	return neg ? (r ? m - r : 0) : r;
}

static uint64_t invmod(uint64_t a, uint64_t m)  // m need not be prime; gcd(a,m) = 1
{
	int64_t t = 0, nt = 1; int64_t r = (int64_t)m, nr = (int64_t)(a % m);
	while (nr) { int64_t q = r / nr, tmp = t - q * nt; t = nt; nt = tmp; tmp = r - q * nr; r = nr; nr = tmp; }
	return (uint64_t)(t < 0 ? t + (int64_t)m : t);
}

// The sieved side: F(a,b) = sum c[k] a^k b^(d-k) (rational side: c = {Y0, Y1}), as decimal strings.
static std::vector<std::string> g_spoly;

// ---------------------------------------------------------------- polynomials over F_p (p < 2^63), for the roots of F
struct FPol { int deg = -1; uint64_t c[2 * MAXD + 1]; };     // coefficients low to high; deg -1: the zero polynomial
struct Fp {
	uint64_t p;
	uint64_t mul(uint64_t a, uint64_t b) const { return p < (1ull << 32) ? a * b % p : (uint64_t)((u128)a * b % p); }
	uint64_t add(uint64_t a, uint64_t b) const { uint64_t s = a + b; return s >= p ? s - p : s; }
	uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + (p - b); }
	void trim(FPol& f) const { while (f.deg >= 0 && f.c[f.deg] == 0) f.deg--; }
	void monic(FPol& f) const {
		if (f.deg < 0) return;
		uint64_t i = invmod(f.c[f.deg], p);
		for (int k = 0; k <= f.deg; k++) f.c[k] = mul(f.c[k], i);
	}
	void rem(FPol& r, const FPol& m) const {              // r mod m, m monic
		for (int k = r.deg; k >= m.deg; k--) {
			uint64_t t = r.c[k];
			if (t) for (int s = 0; s <= m.deg; s++) r.c[k - m.deg + s] = sub(r.c[k - m.deg + s], mul(t, m.c[s]));
		}
		if (r.deg >= m.deg) r.deg = m.deg - 1;
		trim(r);
	}
	FPol mulmod(const FPol& a, const FPol& b, const FPol& m) const {
		FPol r;
		if (a.deg < 0 || b.deg < 0) return r;
		r.deg = a.deg + b.deg;
		for (int k = 0; k <= r.deg; k++) r.c[k] = 0;
		for (int x = 0; x <= a.deg; x++) if (a.c[x]) for (int y = 0; y <= b.deg; y++) r.c[x + y] = add(r.c[x + y], mul(a.c[x], b.c[y]));
		rem(r, m);
		return r;
	}
	FPol powmod(const FPol& b, uint64_t e, const FPol& m) const {
		FPol r; r.deg = 0; r.c[0] = 1;
		if (m.deg == 0) { r.deg = -1; return r; }
		for (int bit = 63 - __builtin_clzll(e); bit >= 0; bit--) { r = mulmod(r, r, m); if (e >> bit & 1) r = mulmod(r, b, m); }
		return r;
	}
	FPol gcd(FPol a, FPol b) const {
		trim(a); trim(b);
		while (b.deg >= 0) { monic(b); rem(a, b); std::swap(a, b); }
		monic(a);
		return a;
	}
	FPol divexact(FPol a, const FPol& m) const {          // a / m, m monic and dividing a
		FPol q; q.deg = a.deg - m.deg;
		for (int k = a.deg; k >= m.deg; k--) {
			uint64_t t = a.c[k]; q.c[k - m.deg] = t;
			if (t) for (int s = 0; s <= m.deg; s++) a.c[k - m.deg + s] = sub(a.c[k - m.deg + s], mul(t, m.c[s]));
		}
		return q;
	}
	void split(const FPol& g, std::vector<uint64_t>& out) const {   // g monic, a product of distinct linear factors; p odd
		if (g.deg <= 0) return;
		if (g.deg == 1) { out.push_back(g.c[0] ? p - g.c[0] : 0); return; }
		for (uint64_t a = 1;; a++) {                      // Cantor-Zassenhaus: gcd(g, (x+a)^((p-1)/2) - 1)
			FPol b; b.deg = 1; b.c[0] = a % p; b.c[1] = 1;
			FPol w = powmod(b, (p - 1) / 2, g);
			if (w.deg < 0) { w.deg = 0; w.c[0] = p - 1; } else w.c[0] = sub(w.c[0], 1);
			trim(w);
			FPol h = gcd(g, w);
			if (h.deg > 0 && h.deg < g.deg) { split(h, out); split(divexact(g, h), out); return; }
		}
	}
};

// distinct roots in F_p of the polynomial with coefficients f[0..d] (reduced mod p), ascending; false if it is 0 mod p
static bool roots_mod_p(const uint64_t* f, int d, uint64_t p, std::vector<uint64_t>& out)
{
	out.clear();
	Fp F{p}; FPol g; g.deg = d;
	for (int k = 0; k <= d; k++) g.c[k] = f[k];
	F.trim(g);
	if (g.deg < 0) return false;
	if (g.deg == 0) return true;
	if (p < 2048) {
		for (uint64_t x = 0; x < p; x++) {
			uint64_t v = 0;
			for (int k = g.deg; k >= 0; k--) v = (v * x + g.c[k]) % p;
			if (v == 0) out.push_back(x);
		}
		return true;
	}
	F.monic(g);
	FPol x; x.deg = 1; x.c[0] = 0; x.c[1] = 1;
	FPol h = F.powmod(x, p, g);                           // x^p - x mod g: its gcd with g is the product of the x - r
	for (int k = h.deg + 1; k <= 1; k++) h.c[k] = 0;
	if (h.deg < 1) h.deg = 1;
	h.c[1] = F.sub(h.c[1], 1);
	F.trim(h);
	F.split(F.gcd(g, h), out);
	std::sort(out.begin(), out.end());
	return true;
}

// the sieved polynomial mod m
static void spoly_mod(uint64_t m, uint64_t* f)
{
	for (size_t k = 0; k < g_spoly.size(); k++) f[k] = mod_str(g_spoly[k], m);
}

// roots of the sieved polynomial modulo the prime q (the special-q of side 1): affine roots only (q | c_d: no projective q)
static std::vector<uint64_t> spoly_roots(uint64_t q)
{
	uint64_t f[MAXD + 1]; spoly_mod(q, f);
	std::vector<uint64_t> r;
	if (!roots_mod_p(f, (int)g_spoly.size() - 1, q, r)) r.clear();
	return r;
}

// roots modulo m = p^k (m < 2^20) of the polynomial f[0..d] (reduced mod m) that are = some root of prev modulo m/p
static std::vector<uint64_t> lift_roots(const uint64_t* f, int d, uint64_t m, uint64_t p, const std::vector<uint64_t>& prev)
{
	std::vector<uint64_t> out;
	const uint64_t step = m / p;
	for (uint64_t r : prev)
		for (uint64_t t = 0; t < p; t++) {
			uint64_t x = r + t * step, v = 0;
			for (int k = d; k >= 0; k--) v = (v * x + f[k]) % m;
			if (v == 0) out.push_back(x);
		}
	std::sort(out.begin(), out.end());
	return out;
}

// root of a linear sieved side c1*x + c0 modulo a prime q not dividing c1: x = -c0/c1
static uint64_t lin_root(uint64_t m)
{
	uint64_t y0 = mod_str(g_spoly[0], m), y1 = mod_str(g_spoly[1], m);
	uint64_t r = y0 ? m - y0 : 0;
	return (uint64_t)((u128)r * invmod(y1, m) % m);
}

// primes in [lo, hi) (2 excluded), by a segmented sieve
static std::vector<uint64_t> primes_in(uint64_t lo, uint64_t hi)
{
	std::vector<uint64_t> out;
	if (hi <= lo) return out;
	const uint64_t root = (uint64_t)sqrtl((long double)hi) + 2;
	std::vector<uint8_t> comp(root + 1, 0); std::vector<uint64_t> sp;
	for (uint64_t p = 2; p <= root; p++) { if (comp[p]) continue; sp.push_back(p); for (uint64_t m = p * p; m <= root; m += p) comp[m] = 1; }
	const uint64_t W = 1 << 22;
	std::vector<uint8_t> seg(W);
	for (uint64_t s = lo; s < hi; s += W) {
		const uint64_t e = std::min(hi, s + W);
		std::fill(seg.begin(), seg.end(), 0);
		for (uint64_t p : sp) {
			if (p * p >= e) break;
			for (uint64_t m = std::max(p * p, (s + p - 1) / p * p); m < e; m += p) seg[m - s] = 1;
		}
		for (uint64_t x = s; x < e; x++) if (!seg[x - s] && x > 2) out.push_back(x);
	}
	return out;
}

// body(k) for k = 0 .. n-1 on nthreads threads (blocks of 64 indices)
static void parallel_for(size_t n, int nthreads, const std::function<void(size_t)>& body)
{
	nthreads = (int)std::max<size_t>(1, std::min<size_t>(nthreads, (n + 63) / 64));
	std::atomic<size_t> next{0};
	auto work = [&]() { for (;;) { size_t s = next.fetch_add(64); if (s >= n) break; for (size_t k = s; k < std::min(n, s + 64); k++) body(k); } };
	std::vector<std::thread> th;
	for (int t = 1; t < nthreads; t++) th.emplace_back(work);
	work();
	for (auto& t : th) t.join();
}

struct QLattice { uint64_t q, r; int64_t a0, b0, a1, b1; uint32_t J; };

// ---------------------------------------------------------------- las' choice of J
// F(a,b) = sum c[k] a^k b^(d-k) of one side as doubles; sq: the special-q side (norms divided by q)
struct Side { std::vector<double> c; bool sq; };
static std::vector<Side> g_sides;                 // set in main from the poly file

static double peval(const std::vector<double>& p, double x) { double v = 0; for (int k = (int)p.size() - 1; k >= 0; k--) v = v * x + p[k]; return v; }
static std::vector<double> deriv(const std::vector<double>& p) { std::vector<double> d; for (size_t k = 1; k < p.size(); k++) d.push_back(k * p[k]); return d; }

// real roots of p (coefficients low to high), ascending: the roots of p' split the line into monotonic pieces, bisected
static std::vector<double> real_roots(std::vector<double> p)
{
	while (!p.empty() && p.back() == 0) p.pop_back();
	const int n = (int)p.size() - 1;
	if (n < 1) return {};
	if (n == 1) return {-p[0] / p[1]};
	std::vector<double> pts = real_roots(deriv(p));
	double B = 0;                                     // Fujiwara's bound on the roots
	for (int k = 0; k < n; k++) B = std::max(B, pow(fabs(p[k] / p[n]) / (k == 0 ? 2 : 1), 1.0 / (n - k)));
	B = 2 * B + 1;
	for (double x : pts) B = std::max(B, fabs(x) + 1);
	pts.insert(pts.begin(), -B); pts.push_back(B);
	std::vector<double> r;
	for (size_t t = 0; t + 1 < pts.size(); t++) {
		double lo = pts[t], hi = pts[t + 1], flo = peval(p, lo), fhi = peval(p, hi);
		if (flo == 0) { if (r.empty() || r.back() != lo) r.push_back(lo); continue; }
		if (fhi == 0 || (flo < 0) == (fhi < 0)) continue;
		for (int it = 0; it < 2200; it++) {
			double mid = 0.5 * (lo + hi);
			if (mid == lo || mid == hi) break;
			double fm = peval(p, mid);
			if (fm == 0) { lo = hi = mid; break; }
			if ((fm < 0) == (flo < 0)) lo = mid; else hi = mid;
		}
		r.push_back(0.5 * (lo + hi));
	}
	return r;
}

// las get_maxnorm_rectangular: max |F(x,y)| over -X <= x <= X, 0 <= y <= Y for a homogeneous F given by F(x,1); the
// critical points of F(x,1) and F(1,y) do not depend on X, Y and are found once
struct RectNorm {
	std::vector<double> p, rev, cp, crev; double d;
	explicit RectNorm(const std::vector<double>& f) : p(f), rev(f.rbegin(), f.rend()), d((double)f.size() - 1) {
		cp = real_roots(deriv(p)); crev = real_roots(deriv(rev));
	}
	static double aux(const std::vector<double>& q, const std::vector<double>& crit, double s) {   // max |q(x)|, |x| <= s
		if (q.size() == 1) return fabs(q[0]);
		double m = std::max(fabs(peval(q, -s)), fabs(peval(q, s)));
		for (double x : crit) if (fabs(x) <= s) m = std::max(m, fabs(peval(q, x)));
		return m;
	}
	double operator()(double X, double Y) const { return std::max(aux(p, cp, X / Y) * pow(Y, d), aux(rev, crev, Y / X) * pow(X, d)); }
};

// las get_maxnorm_circular: max |F(X cos t, Y sin t)| for 0 <= t <= pi
static double maxnorm_circ(const std::vector<double>& f, double X, double Y)
{
	const int d = (int)f.size() - 1;
	std::vector<double> P(d + 2, 0.0);
	for (int i = 1; i <= d; i++) { double t = pow(X / Y, i - 1.0); P[i + 1] += X * i * f[i] * t; P[i - 1] += X * i * f[i] * t; }
	for (int i = 0; i <= d; i++) P[i + 1] -= d * Y * f[i] * pow(X / Y, (double)i);
	P[d + 1] = 0;           // cancels exactly (las leaves the rounding residue, whose huge root only gives F(+-X, ~0))
	double m = fabs(f[d] * pow(X, (double)d));
	for (double u : real_roots(P)) {
		double x = X * u / sqrt(1 + u * u), y = Y / sqrt(1 + u * u), v = 0;
		for (int k = d; k >= 0; k--) v = v * x + f[k] * pow(y, d - k);
		m = std::max(m, fabs(v));
	}
	return m;
}

// F(a0*i + a1*j, b0*i + b1*j) as a polynomial in i (j = 1): coefficient k of i^k j^(d-k)
template <typename T> static std::vector<T> homography(const std::vector<T>& c, T a0, T b0, T a1, T b1)
{
	const int d = (int)c.size() - 1;
	std::vector<std::vector<T>> A(d + 1), B(d + 1);    // powers of a0*x + a1 and b0*x + b1
	A[0] = B[0] = {1};
	for (int k = 1; k <= d; k++) {
		A[k].assign(k + 1, 0); B[k].assign(k + 1, 0);
		for (int t = 0; t < k; t++) { A[k][t] += A[k - 1][t] * a1; A[k][t + 1] += A[k - 1][t] * a0; B[k][t] += B[k - 1][t] * b1; B[k][t + 1] += B[k - 1][t] * b0; }
	}
	std::vector<T> r(d + 1, 0);
	for (int m = 0; m <= d; m++)
		for (int x = 0; x <= m; x++) for (int y = 0; y <= d - m; y++) r[x + y] += c[m] * A[m][x] * B[d - m][y];
	return r;
}

// CADO generic_skew_gauss (quadratic form a^2 + s^2 b^2 with s^2 = mz * 2^ex) followed by sieve_info_adjust_IJ (strategy 0).

static QLattice reduce_q(uint64_t q, uint64_t r, double skew, int logI)
{
	int ex; double mant = frexp(skew * skew, &ex);
	int64_t mz = (int64_t)ldexp(mant, 53); ex -= 53;
	i128 a[2] = {(i128)q, (i128)r}, b[2] = {0, 1};
	auto quad = [&](int x, int y) -> i128 {
		i128 d = a[x] * a[y], t = b[x] * b[y];
		if (ex < 0) d <<= -ex;
		if (ex > 0) t <<= ex;
		return d + t * mz;
	};
	auto rdiv = [](i128 n, i128 d) -> i128 {   // round to nearest, like mpz_rdiv_q
		bool neg = (n < 0) != (d < 0);
		i128 an = n < 0 ? -n : n, ad = d < 0 ? -d : d;
		i128 qq = (2 * an + ad) / (2 * ad);
		return neg ? -qq : qq;
	};
	i128 N0 = quad(0, 0), N1 = quad(1, 1), S = quad(0, 1);
	for (;;) {
		i128 k = rdiv(S, N1);
		if (k == 0) break;
		a[0] -= k * a[1]; b[0] -= k * b[1];
		i128 tmp = S; S -= k * N1; tmp += S; N0 -= k * tmp;
		k = rdiv(S, N0);
		if (k == 0) break;
		a[1] -= k * a[0]; b[1] -= k * b[0];
		tmp = S; S -= k * N0; tmp += S; N1 -= k * tmp;
	}
	if (N0 > N1) { std::swap(a[0], a[1]); std::swap(b[0], b[1]); }
	QLattice L{q, r, (int64_t)a[0], (int64_t)b[0], (int64_t)a[1], (int64_t)b[1], 0};
	// adjust_IJ: u0 must be the smaller in the skewed L2 norm a^2/s + b^2 s
	auto sn = [&](int64_t x, int64_t y) { return (double)x * x / skew + (double)y * y * skew; };
	if (sn(L.a0, L.b0) > sn(L.a1, L.b1)) { std::swap(L.a0, L.a1); std::swap(L.b0, L.b1); }
	double rt = sqrt(skew);
	double maxab1 = std::max(fabs((double)L.a1) / rt, fabs((double)L.b1) * rt);
	double B = sqrt(2.0 * (double)q / sqrt(3.0));
	(void)maxab1; (void)B;   // las computes a J from these (sieve_info_adjust_IJ) but then replaces it, see below
	// las sieve_info_update_norm_data_Jmax: the largest value |F(i,j)| over |i| <= I/2, 0 <= j < J must not exceed the
	// largest value over an ideal (elliptic) region of the same area, times a fudge 2 (closed forms for linear sides).
	const double fudge = 2.0, Id = (double)(1u << logI), qd = (double)q;
	const double Aw = Id * sqrt(qd * skew), Bw = (double)(1u << (logI - 1)) * sqrt(qd / skew);
	double Jmax = (double)(1u << (logI - 1));
	for (const Side& S : g_sides) {
		double X = fudge * Aw / 2, Y = fudge * Bw;
		std::function<double(double)> v;
		double maxnorm;
		if (S.c.size() == 2) {
			const double c1 = S.c[1], c0 = S.c[0];
			maxnorm = sqrt((c1 * X) * (c1 * X) + (c0 * Y) * (c0 * Y));
			double alpha = c1 * (double)L.a0 + c0 * (double)L.b0, beta = c1 * (double)L.a1 + c0 * (double)L.b1;
			if (S.sq) { maxnorm /= qd; alpha /= qd; beta /= qd; }
			v = [=](double J) { return fabs(alpha) * Id / 2 + fabs(beta) * J; };
		} else {
			maxnorm = maxnorm_circ(S.c, X, Y);
			std::vector<double> fij = homography<double>(S.c, (double)L.a0, (double)L.b0, (double)L.a1, (double)L.b1);
			if (S.sq) { maxnorm /= qd; for (double& x : fij) x /= qd; }
			RectNorm R(fij);
			v = [=](double J) { return R(Id / 2, J); };
		}
		if (v(Jmax) > maxnorm) {
			double a = 0, b = Jmax;
			while (trunc(a) != trunc(b)) { double c = (a + b) * 0.5; if (v(c) < maxnorm) a = c; else b = c; }
			Jmax = trunc(a) + 1;
		}
	}
	uint32_t J = (uint32_t)Jmax;
	uint32_t lines = 1u << std::max(16 - logI, 0);       // LOG_BUCKET_REGION = 16: round down to whole bucket regions
	J = (J / lines) * lines;
	L.J = J;
	return L;
}

// ---------------------------------------------------------------------------------------------------------------- device

struct DevFB {
	uint32_t* p; uint32_t* base; uint32_t* y0; uint32_t* y1; uint8_t* logp;   // all entries (small first): modulus p^k, prime p, Y0 and Y1 mod p^k
	uint32_t* R;                                // transformed root i = j*R (mod p) for this special-q; flags below
	uint32_t* proj; uint32_t* nproj;            // the small entries with R = R_PROJ for this special-q (usually a few)
	uint4* part; uint32_t* npart;               // small prime powers hitting only rows j = 0 mod p^e: (p^k/p^e, R', p^e, logp)
	int n, nsmall;
};
#ifndef CPAD
#define CPAD 1                  // stride of the row counters (32 = one per 128-byte line: slower in tests)
#endif
#define R_SKIP 0xffffffffu      // no hits in the sieve region (p == q, vertical line, ...)
#define R_PROJ 0xfffffffeu      // projective: p divides every value of the rows j = 0 (mod p)

__device__ __forceinline__ uint32_t d_mod(int64_t x, uint32_t p) { int64_t r = x % (int64_t)p; return (uint32_t)(r < 0 ? r + p : r); }
__device__ uint32_t d_invmod(uint32_t a, uint32_t m)
{
	int64_t t = 0, nt = 1, r = m, nr = a;
	while (nr) { int64_t q = r / nr, tmp = t - q * nt; t = nt; nt = tmp; tmp = r - q * nr; r = nr; nr = tmp; }
	return (uint32_t)(t < 0 ? t + m : t);
}

// transformed roots. Each entry stands for a linear form y1*a + y0*b whose multiples of p (= p^k) are its hits: on the
// rational side (y1, y0) = (Y1, Y0) mod p, needing no root (so primes dividing Y1 are handled too); on a side of degree >= 2
// (1, -r) for an affine root r of F mod p (a = r*b) and (-s, 1) for a projective one (b = s*a, s = 0 mod the prime).
// p | y1*a + y0*b with (a,b) = i*(a0,b0)+j*(a1,b1)  <=>  i*A + j*B = 0 (mod p), A = y1*a0 + y0*b0, B = y1*a1 + y0*b1.
__global__ void k_transform(DevFB fb, uint64_t q, int64_t a0, int64_t b0, int64_t a1, int64_t b1)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= fb.n) return;
	uint32_t p = fb.p[k];
	uint64_t y0 = fb.y0[k], y1 = fb.y1[k];
	if (p == q || (q % p == 0 && k < fb.nsmall)) { fb.R[k] = R_SKIP; return; }
	uint64_t A = (y1 * d_mod(a0, p) % p + y0 * d_mod(b0, p) % p) % p;
	uint64_t B = (y1 * d_mod(a1, p) % p + y0 * d_mod(b1, p) % p) % p;
	if (A == 0) {   // p^k | i*A + j*B  <=>  p^k | j (when p does not divide B): whole rows j = 0 (mod p^k)
		if (k < fb.nsmall && B % fb.base[k] != 0) { fb.R[k] = R_PROJ; fb.proj[atomicAdd(fb.nproj, 1u)] = k; }
		else fb.R[k] = R_SKIP;
		return;
	}
	// for prime powers A must be invertible mod p^k: A != 0 mod p (otherwise the hits are not a simple progression: skipped)
	uint32_t inv = d_invmod((uint32_t)A, p);
	if ((uint64_t)inv * A % p != 1) {
		// p^k with p^e || A, 0 < e < k (and p not dividing B): the hits are in the rows j = p^e * j', at
		// i = j' * R' mod p^(k-e) with R' = -B / (A / p^e): entered in the list of partial entries
		const uint32_t base = fb.base[k];
		if (k < fb.nsmall && B % base != 0) {
			uint32_t pe = 1, Ar = (uint32_t)A;
			while (Ar % base == 0) { Ar /= base; pe *= base; }
			const uint32_t m2 = p / pe;
			const uint32_t R2 = (uint32_t)((uint64_t)((m2 - (uint32_t)(B % m2)) % m2) * d_invmod(Ar % m2, m2) % m2);
			fb.part[atomicAdd(fb.npart, 1u)] = make_uint4(m2, R2, pe, fb.logp[k]);
		}
		fb.R[k] = R_SKIP; return;
	}
	fb.R[k] = (uint32_t)((uint64_t)(B ? p - B : 0) * inv % p);
}

// Franke-Kleinjung: reduce the lattice {(i,j): i = j*R mod p} for width I to warp (-mi0,j0), step (i1,j1).
__device__ bool fk_reduce(uint32_t p, uint32_t R, uint32_t I, uint32_t& mi0, uint32_t& j0, uint32_t& i1, uint32_t& j1)
{
	if (R == 0) return false;                  // vertical line i = 0: only non-coprime positions
	mi0 = p; j0 = 0; i1 = R; j1 = 1;
	for (;;) {
		if (i1 < I) {
			if (i1 == 0) return false;
			uint32_t a = (mi0 + i1 - I) / i1;
			mi0 -= a * i1; j0 += a * j1;
			return true;
		}
		{ uint32_t k = mi0 / i1; mi0 -= k * i1; j0 += k * j1; }
		if (mi0 < I) {
			if (mi0 == 0) return false;
			uint32_t a = (mi0 + i1 - I) / mi0;
			i1 -= a * mi0; j1 += a * j0;
			return true;
		}
		{ uint32_t k = i1 / mi0; i1 -= k * mi0; j1 += k * j0; }
	}
}

// Bucket regions: a row of I = 2^logI columns is split into regions of 2^logR columns, logR = min(logI, 15), so that a
// region fits into the shared memory of a block (32 KB) whatever I is. A special-q is sieved in one or more passes over its
// rows (-parts); in the pass over the rows jlo <= j < jhi, region g = (j - jlo) * (I >> logR) + (x >> logR).
// large primes: walk their hits in the strip in increasing j and append (column in the region, logp) to the bucket of the
// region. With several passes, st keeps each prime's next hit (x, j) from one pass to the next (resume: continue from it).
__device__ __forceinline__ void fk_step(uint32_t& x, uint32_t& j, uint32_t I, uint32_t mi0, uint32_t j0, uint32_t i1, uint32_t j1)
{
	if (x >= mi0) { x -= mi0; j += j0; }
	else if (x < I - i1) { x += i1; j += j1; }
	else { x += i1 - mi0; j += j0 + j1; }
}
__global__ void k_buckets(DevFB fb, uint32_t logI, uint32_t logR, uint32_t jlo, uint32_t jhi, uint2* st, bool resume,
                          uint32_t* bucket, uint32_t* count, uint32_t cap, unsigned long long* dropped)
{
	const uint32_t I = 1u << logI, shift = logI - logR, rmask = (1u << logR) - 1;
	for (int k = fb.nsmall + blockIdx.x * blockDim.x + threadIdx.x; k < fb.n; k += gridDim.x * blockDim.x) {
		uint32_t R = fb.R[k];
		if (R == R_SKIP) continue;
		uint32_t p = fb.p[k];
		uint32_t mi0, j0, i1, j1;
		if (!fk_reduce(p, R, I, mi0, j0, i1, j1)) continue;
		uint32_t lp = fb.logp[k];
		uint32_t x, j;
		if (resume) { const uint2 s = st[k - fb.nsmall]; x = s.x; j = s.y; }
		else { x = I >> 1; j = 0; fk_step(x, j, I, mi0, j0, i1, j1); }     // x = i + I/2: the first hit after (0,0)
		while (j < jhi) {
			const size_t g = ((size_t)(j - jlo) << shift) | (x >> logR);
			uint32_t slot = atomicAdd(&count[g * CPAD], 1u);
			if (slot < cap) bucket[g * cap + slot] = ((x & rmask) << 8) | lp;
			else atomicAdd(dropped, 1ull);
			fk_step(x, j, I, mi0, j0, i1, j1);
		}
		if (st) st[k - fb.nsmall] = make_uint2(x, j);
	}
}


// Large primes, two passes per block over a chunk of primes: count the hits per row in shared memory, reserve each row's
// space in the global bucket with one atomic per row, then walk again and write the entries into place.
__device__ __forceinline__ bool fk_next(uint32_t& x, uint32_t& j, uint32_t I, uint32_t mi0, uint32_t j0, uint32_t i1, uint32_t j1)
{
	if (x >= mi0) { x -= mi0; j += j0; }
	else if (x < I - i1) { x += i1; j += j1; }
	else { x += i1 - mi0; j += j0 + j1; }
	return true;
}
__global__ void __launch_bounds__(1024, 1)
k_buckets2(DevFB fb, uint32_t logI, uint32_t J, const int* chunk, uint32_t* bucket, uint32_t* count, uint32_t cap,
           unsigned long long* dropped)
{
	extern __shared__ uint32_t hist[];           // J counters
	const uint32_t I = 1u << logI;
	const int k0 = chunk[blockIdx.x], k1 = chunk[blockIdx.x + 1];
	for (uint32_t r = threadIdx.x; r < J; r += blockDim.x) hist[r] = 0;
	__syncthreads();
	for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
		uint32_t R = fb.R[k];
		if (R >= R_PROJ) continue;
		uint32_t mi0, j0, i1, j1;
		if (!fk_reduce(fb.p[k], R, I, mi0, j0, i1, j1)) continue;
		uint32_t x = I >> 1, j = 0;
		for (;;) { fk_next(x, j, I, mi0, j0, i1, j1); if (j >= J) break; atomicAdd(&hist[j], 1u); }
	}
	__syncthreads();
	for (uint32_t r = threadIdx.x; r < J; r += blockDim.x) {
		uint32_t n = hist[r];
		if (n) hist[r] = atomicAdd(&count[(size_t)r * CPAD], n);    // hist becomes this block's write cursor in row r
	}
	__syncthreads();
	for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
		uint32_t R = fb.R[k];
		if (R >= R_PROJ) continue;
		uint32_t mi0, j0, i1, j1;
		if (!fk_reduce(fb.p[k], R, I, mi0, j0, i1, j1)) continue;
		uint32_t lp = fb.logp[k], x = I >> 1, j = 0;
		for (;;) {
			fk_next(x, j, I, mi0, j0, i1, j1);
			if (j >= J) break;
			uint32_t slot = atomicAdd(&hist[j], 1u);
			if (slot < cap) bucket[(size_t)j * cap + slot] = (x << 8) | lp;
			else atomicAdd(dropped, 1ull);
		}
	}
}

__device__ __forceinline__ void add_log(uint32_t* sh, uint32_t x, uint32_t lp) { atomicAdd(&sh[x >> 2], lp << ((x & 3) * 8)); }

// Variants of the row kernel: one region per row (I <= 2^15), several (I = 2^16), several with j*R above 2^32 (I >= 2^17).
enum { ONE_REGION = 0, SPLIT32 = 1, SPLIT_WIDE = 2 };
// first column >= x0 of row j hit by the entry (p, R), relative to x0 (i = j*R mod p, column x = i + I/2)
template <int MODE> __device__ __forceinline__ uint32_t region_start(uint32_t j, uint32_t R, uint32_t p, uint32_t I, uint32_t x0)
{
	if (MODE == ONE_REGION) return (j * R + (I >> 1)) % p;                  // j < I/2, R < p < I <= 2^15
	if (MODE == SPLIT32) return (j * R + (I >> 1) + p - x0 % p) % p;        // I = 2^16: still below 2^32
	// I = 2^17 .. 2^19: j < I/2 and R < p < I, so j*R needs up to 37 bits; 64-bit division is very slow on the GPU, so j is
	// split into bytes and every intermediate value stays below 2^29
	uint32_t jr = (((j >> 8) * R) % p * 256 + (j & 255) * R) % p;
	return (jr + (I >> 1) + p - x0 % p) % p;
}

__device__ uint32_t gcd32(uint32_t a, uint32_t b)
{
	if (!a) return b; if (!b) return a;
	int s = __ffs(a | b) - 1; a >>= __ffs(a) - 1;
	do { b >>= __ffs(b) - 1; if (a > b) { uint32_t t = a; a = b; b = t; } b -= a; } while (b);
	return a << s;
}

struct OutPair { int64_t a; int64_t b; };

// one block per region (blocks loop over the regions when the grid is smaller): small primes into shared memory, then the
// region's bucket, then the survivors. 3 blocks of 512 threads per SM: the shared memory (32 KB per block for I = 2^15) allows no more
// on these GPUs, and sm_86/89/120 have at most 1536 threads per SM (sm_75: 1024).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
#define ROWS_MINB 2
#else
#define ROWS_MINB 3
#endif
// Norm of a sieved side of degree d >= 2: F(i,j)/q = 2^log_scale * sum G[k] t^k u^(d-k) with t = i/(I/2), u = j/(I/2), so
// |t|, u <= 1 and |G[k]| <= 1 keep the float values in range.
struct NormPoly { float G[MAXD + 1]; int d; float invS; };

template <int MODE, bool POLY> __global__ void __launch_bounds__(512, ROWS_MINB)
k_rows(DevFB fb, uint32_t logI, uint32_t logR, uint32_t jlo, uint32_t jn, const uint32_t* bucket, const uint32_t* count, uint32_t cap,
                       int64_t a0, int64_t b0, int64_t a1, int64_t b1, float alpha, float beta, NormPoly np, float log_scale,
                       float mfb_slack, bool dummy, long long c1, long long c0, float dummy_bits,
                       OutPair* out, unsigned int* nout, unsigned int maxout, uint32_t tiny)
{
	extern __shared__ uint32_t sh[];
	const uint32_t I = 1u << logI, RS = 1u << logR, shift = MODE == ONE_REGION ? 0 : logI - logR, rpr = 1u << shift;
	// No static shared memory: with 96 KB of shared memory per SM (sm_70), 3 blocks of exactly 32 KB fill it, and 16 bytes more (shared
	// memory is allocated in units of 256 bytes) left room for only 2 blocks.
	const uint32_t nproj = *fb.nproj, npart = *fb.npart;
	// regions of the rows jlo .. jlo+jn-1 of this pass (row 0 is not sieved); g is the region's index in the pass' buckets
	for (uint32_t g = blockIdx.x + (jlo ? 0 : rpr); g < jn * rpr; g += gridDim.x) {
	const uint32_t j = jlo + (g >> shift), x0 = MODE == ONE_REGION ? 0 : (g & (rpr - 1)) << logR;   // row, first column of the region
	__syncthreads();                // the previous region's survivors are read
	for (uint32_t w = threadIdx.x; w < RS / 4; w += blockDim.x) sh[w] = 0;
	uint32_t bonus = 0;             // log sum of the projective entries dividing every value of this row
	for (uint32_t t = 0; t < nproj; t++) { int k = fb.proj[t]; if (j % fb.p[k] == 0) bonus += fb.logp[k]; }
	__syncthreads();
	// tiny primes (p < tiny, default 64): all threads share each prime. Larger primes have too few hits in a region for all
	// threads to share them (in tests, the rows took twice as long with tiny 1024 as with 64)
	for (int k = 0; k < fb.nsmall; k++) {
		uint32_t p = fb.p[k];
		if (p >= tiny) break;
		uint32_t R = fb.R[k];
		if (R >= R_PROJ) continue;
		uint32_t start = region_start<MODE>(j, R, p, I, x0), lp = fb.logp[k];
		for (uint32_t x = start + threadIdx.x * p; x < RS; x += blockDim.x * p) add_log(sh, x, lp);
	}
	// partial entries (prime powers whose hits are only in the rows j = 0 mod p^e): all threads share each one
	for (uint32_t t = 0; t < npart; t++) {
		const uint4 e = fb.part[t];
		if (j % e.z) continue;
		const uint32_t m2 = e.x, start = region_start<MODE>(j / e.z, e.y, m2, I, x0);
		for (uint32_t x = start + threadIdx.x * m2; x < RS; x += blockDim.x * m2) add_log(sh, x, e.w);
	}
	// other small primes: one thread each (entries are sorted by p; prime powers of tiny primes are also tiny)
	for (int k = threadIdx.x; k < fb.nsmall; k += blockDim.x) {
		uint32_t p = fb.p[k];
		if (p < tiny) continue;
		uint32_t R = fb.R[k];
		if (R >= R_PROJ) continue;
		uint32_t lp = fb.logp[k];
		for (uint32_t x = region_start<MODE>(j, R, p, I, x0); x < RS; x += p) add_log(sh, x, lp);
	}
	// large primes from the bucket of this region
	uint32_t n = min(count[(size_t)g * CPAD], cap);
	const uint32_t* bk = bucket + (size_t)g * cap;
	for (uint32_t k = threadIdx.x; k < n; k += blockDim.x) { uint32_t e = bk[k]; add_log(sh, e >> 8, e & 255); }
	__syncthreads();
	// survivors
	const uint8_t* bytes = (const uint8_t*)sh;
	const float bj = beta * (float)j - alpha * ((float)(I >> 1) - (float)x0);   // value/q = alpha*i + beta*j, i = x0 + x - I/2
	float hc[MAXD + 1], t0 = 0;     // POLY: the row's polynomial in t, coefficient k = G[k] u^(d-k)
	if (POLY) {
		const float u = (float)j * np.invS;
		float pw = 1.f;
#pragma unroll
		for (int k = MAXD; k >= 0; k--) { if (k <= np.d) { hc[k] = np.G[k] * pw; pw *= u; } else hc[k] = 0.f; }
		t0 = ((float)x0 - (float)(I >> 1)) * np.invS;
	}
	for (uint32_t x = threadIdx.x; x < RS; x += blockDim.x) {
		float lv;                       // log2 of the value/q scaled by 2^-log_scale
		if (POLY) {
			const float t = fmaf((float)x, np.invS, t0);
			float v = hc[MAXD];
#pragma unroll
			for (int k = MAXD - 1; k >= 0; k--) v = fmaf(v, t, hc[k]);
			v = fabsf(v);
			if (v >= 1e-30f) lv = __log2f(v);
			else {
				// Below about 2^-100 (near the origin or a real root; degree 7 at I = 17: values below about 2^57) the float
				// terms underflow: again in double, so that these small, very smooth values are not lost.
				const double td = ((double)x0 + (double)x - (double)(I >> 1)) * (double)np.invS, ud = (double)j * (double)np.invS;
				double hd[MAXD + 1], pd = 1.0;
#pragma unroll
				for (int k = MAXD; k >= 0; k--) { if (k <= np.d) { hd[k] = (double)np.G[k] * pd; pd *= ud; } else hd[k] = 0.0; }
				double vd = hd[MAXD];
#pragma unroll
				for (int k = MAXD - 1; k >= 0; k--) vd = fma(vd, td, hd[k]);
				lv = (float)log2(fmax(fabs(vd), 1e-300));
			}
		} else lv = __log2f(fmaxf(fabsf(fmaf(alpha, (float)x, bj)), 1e-30f));
		const float thr = lv + log_scale - mfb_slack;   // value/q = 2^(lv + log_scale)
		if ((float)(bytes[x] + bonus) < thr) continue;
		int64_t i = (int64_t)(x0 + x) - (int64_t)(I >> 1);
		int64_t b = i * b0 + (int64_t)j * b1;
		if (b == 0) continue;
		int64_t a = i * a0 + (int64_t)j * a1;
		if (gcd32((uint32_t)(i < 0 ? -i : i), j) != 1) continue;
		if (b < 0) { a = -a; b = -b; }
		if (dummy && log2(fabs((double)c1 * (double)a + (double)c0 * (double)b) + 1) >= dummy_bits) continue;
		unsigned int slot = atomicAdd(nout, 1u);
		if (slot < maxout) { out[slot].a = a; out[slot].b = b; }
	}
	}
}

// ---------------------------------------------------------------------------------------------------------------- host

int main(int argc, char** argv)
{
	const char* polyf = nullptr; const char* outf = nullptr; const char* qfile = nullptr;
	int logI = 15; uint64_t lim = 25000000; int mfb = 56; double slack = 12;
	uint64_t q0 = 0, q1 = 0; long nrandom = 0; uint32_t tiny = 64; int verbose = 0; double dummy_bits = 32; int oldbuckets = 1;
	int lanes = 1, bblocks = 0, rblocks = 0; double progress_s = 60; int side = -1; const char* dumpfb = nullptr; double maxout_m = 16;
	long nparts_opt = 0;
	int nthreads = (int)std::min(16u, std::max(1u, std::thread::hardware_concurrency()));
	for (int k = 1; k < argc; k++) {
		std::string s = argv[k];
		auto nxt = [&]() { if (k + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); exit(1); } return argv[++k]; };
		if (s == "-poly") polyf = nxt(); else if (s == "-out") outf = nxt(); else if (s == "-qfile") qfile = nxt();
		else if (s == "-I") logI = atoi(nxt()); else if (s == "-lim") lim = strtoull(nxt(), 0, 10);
		else if (s == "-mfb") mfb = atoi(nxt()); else if (s == "-slack") slack = atof(nxt());
		else if (s == "-q0") q0 = strtoull(nxt(), 0, 10); else if (s == "-q1") q1 = strtoull(nxt(), 0, 10);
		else if (s == "-random") nrandom = atol(nxt()); else if (s == "-tiny") tiny = atoi(nxt());
		else if (s == "-dummybits") dummy_bits = atof(nxt());
		else if (s == "-twopass") oldbuckets = 0;
		else if (s == "-lanes") lanes = std::max(1, std::min(2, atoi(nxt())));
		else if (s == "-bblocks") bblocks = atoi(nxt()); else if (s == "-rblocks") rblocks = atoi(nxt());
		else if (s == "-v") verbose = 1;
		else if (s == "-progress") progress_s = atof(nxt());
		else if (s == "-side") side = atoi(nxt());
		else if (s == "-threads") nthreads = std::max(1, atoi(nxt()));
		else if (s == "-dumpfb") dumpfb = nxt();
		else if (s == "-maxout") maxout_m = atof(nxt());
		else if (s == "-parts") nparts_opt = atol(nxt());
		else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
	}
	if (!polyf || (!dumpfb && (!outf || (!qfile && (!q0 || !q1))))) { fprintf(stderr, "usage: see the top of gpusieve.cu\n"); return 1; }
	Poly P = read_poly(polyf);
	// the sieved side and the other (linear) side
	if (side < 0) side = P.alg_deg >= 2 ? 1 : 0;
	if (side != 0 && side != 1) { fprintf(stderr, "-side must be 0 or 1\n"); return 1; }
	std::vector<std::string> other;                  // the other side's c0, c1 (empty: none)
	if (side == 0) {
		if (P.Y0s.empty()) { fprintf(stderr, "no Y0 in %s\n", polyf); return 1; }
		if (P.alg_deg >= 2) { fprintf(stderr, "%s: the algebraic side has degree %d; the side that is not sieved must be linear "
		                                      "(sieve it with -side 1)\n", polyf, P.alg_deg); return 1; }
		g_spoly = {P.Y0s, P.Y1s};
		if (P.alg_deg >= 0) other = {P.c[0], P.c[1]};
	} else {
		if (P.alg_deg < 1) { fprintf(stderr, "%s: no algebraic side to sieve\n", polyf); return 1; }
		g_spoly.assign(P.c, P.c + P.alg_deg + 1);
		if (!P.Y0s.empty()) other = {P.Y0s, P.Y1s};
	}
	for (std::string& s : g_spoly) if (s.empty()) s = "0";
	for (std::string& s : other) if (s.empty()) s = "0";
	const int sdeg = (int)g_spoly.size() - 1;
	const bool poly_norm = sdeg >= 2;
	{
		Side S{{}, true}; for (const std::string& s : g_spoly) S.c.push_back((double)strtold(s.c_str(), nullptr));
		g_sides.push_back(S);
		if (!other.empty()) g_sides.push_back({{(double)strtold(other[0].c_str(), nullptr), (double)strtold(other[1].c_str(), nullptr)}, false});
	}
	// the other side is a dummy whose values are checked if its coefficients fit into 64 bits
	bool dummy = false; long long dc0 = 0, dc1 = 0;
	if (!other.empty() && dummy_bits > 0) {
		errno = 0; char *e0, *e1;
		dc0 = strtoll(other[0].c_str(), &e0, 10); dc1 = strtoll(other[1].c_str(), &e1, 10);
		dummy = errno == 0 && *e0 == 0 && *e1 == 0;
	}
	{
		char buf[64]; snprintf(buf, sizeof buf, "a dummy, |value| < 2^%g", dummy_bits);
		fprintf(stderr, "sieving side %d (degree %d) with the special-q on it; side %d: %s\n", side, sdeg, 1 - side,
			other.empty() ? "absent" : dummy ? buf : dummy_bits > 0 ? "not checked (coefficients above 64 bits)" : "not checked (-dummybits 0)");
	}
	if (logI < 10 || logI > 19) { fprintf(stderr, "-I must be between 10 and 19\n"); return 1; }
	const uint32_t I = 1u << logI;
	const uint32_t logR = std::min(logI, 15), RS = 1u << logR, rpr = I >> logR;   // bucket regions: rpr per row, RS columns
	if (!oldbuckets && rpr > 1) { fprintf(stderr, "-twopass needs I <= 15\n"); return 1; }
	auto t_start = std::chrono::steady_clock::now();

	// factor base: primes up to lim; prime powers below I for the small primes
	if (lim >= (1ull << 32)) { fprintf(stderr, "lim must be below 2^32\n"); return 1; }
	std::vector<uint32_t> fbprimes; {
		std::vector<uint8_t> comp(lim + 1, 0);
		for (uint64_t p = 2; p <= lim; p++) {
			if (comp[p]) continue;
			fbprimes.push_back((uint32_t)p);
			for (uint64_t m = p * p; m <= lim; m += p) comp[m] = 1;
		}
	}
	struct E { uint32_t p, base, y0, y1; uint8_t lg; };
	std::vector<E> small, large;
	auto lgp = [](uint64_t p) { return (uint8_t)lround(log2((double)p)); };
	if (!poly_norm) {       // linear: (y1, y0) = (c1, c0) mod p^k
		for (uint32_t p : fbprimes) {
			if (p < I) {
				for (uint64_t pk = p; pk < I; pk *= p)
					small.push_back({(uint32_t)pk, p, (uint32_t)mod_str(g_spoly[0], pk), (uint32_t)mod_str(g_spoly[1], pk), lgp(p)});
			} else large.push_back({p, p, (uint32_t)mod_str(g_spoly[0], p), (uint32_t)mod_str(g_spoly[1], p), lgp(p)});
		}
	} else {                // degree >= 2: one entry per root, (y1, y0) = (1, -r) or (-s, 1) mod p^k
		uint64_t f[MAXD + 1], rv[MAXD + 1];
		std::vector<uint64_t> roots, s;
		for (uint32_t p : fbprimes) {
			if (p >= I) break;
			spoly_mod(p, f);
			if (!roots_mod_p(f, sdeg, p, roots)) { fprintf(stderr, "every value of the sieved side is divisible by %u: not sieved\n", p); continue; }
			const bool proj = f[sdeg] == 0;
			for (uint64_t m = p; m < I && !roots.empty(); m *= p) {      // affine roots mod p^k, lifted from those mod p^(k-1)
				if (m > p) { spoly_mod(m, f); roots = lift_roots(f, sdeg, m, p, roots); }
				for (uint64_t r : roots) small.push_back({(uint32_t)m, p, (uint32_t)((m - r) % m), 1, lgp(p)});
			}
			s.assign(proj ? 1 : 0, 0);                                    // projective: roots s = 0 mod p of the reversed polynomial
			for (uint64_t m = p; m < I && !s.empty(); m *= p) {
				if (m > p) { spoly_mod(m, f); for (int k = 0; k <= sdeg; k++) rv[k] = f[sdeg - k]; s = lift_roots(rv, sdeg, m, p, s); }
				for (uint64_t x : s) small.push_back({(uint32_t)m, p, 1, (uint32_t)((m - x) % m), lgp(p)});
			}
		}
		const size_t first = std::lower_bound(fbprimes.begin(), fbprimes.end(), I) - fbprimes.begin(), nl = fbprimes.size() - first;
		const size_t nchunk = 4096;
		std::vector<std::vector<E>> part(nchunk);
		parallel_for(nchunk, nthreads, [&](size_t c) {
			uint64_t f[MAXD + 1]; std::vector<uint64_t> roots;
			for (size_t t = first + nl * c / nchunk; t < first + nl * (c + 1) / nchunk; t++) {
				const uint32_t p = fbprimes[t];
				spoly_mod(p, f);
				if (!roots_mod_p(f, sdeg, p, roots)) continue;
				for (uint64_t r : roots) part[c].push_back({p, p, (uint32_t)((p - r) % p), 1, lgp(p)});
				if (f[sdeg] == 0) part[c].push_back({p, p, 1, 0, lgp(p)});
			}
		});
		for (const auto& v : part) large.insert(large.end(), v.begin(), v.end());
	}
	// small entries sorted by modulus (prime powers of tiny primes stay tiny)
	std::sort(small.begin(), small.end(), [](const E& x, const E& y) { return x.p < y.p; });
	int nsmall = (int)small.size();
	small.insert(small.end(), large.begin(), large.end());
	if (dumpfb) {           // debug: the factor base as "modulus prime y1 y0" (hits: modulus | y1*a + y0*b)
		FILE* d = fopen(dumpfb, "w"); if (!d) { perror(dumpfb); return 1; }
		for (const E& e : small) fprintf(d, "%u %u %u %u\n", e.p, e.base, e.y1, e.y0);
		fclose(d);
		fprintf(stderr, "factor base: %zu entries (%d below I) written to %s in %.1f s\n", small.size(), nsmall, dumpfb,
			std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
		return 0;
	}
	std::vector<uint32_t> fp, fbase, f0, f1; std::vector<uint8_t> fl;
	for (const E& e : small) { fp.push_back(e.p); fbase.push_back(e.base); f0.push_back(e.y0); f1.push_back(e.y1); fl.push_back(e.lg); }
	DevFB fb; fb.n = (int)fp.size(); fb.nsmall = nsmall; fb.proj = fb.nproj = fb.npart = nullptr; fb.part = nullptr;
	CHECK(cudaMalloc(&fb.p, fb.n * 4)); CHECK(cudaMalloc(&fb.base, fb.n * 4)); CHECK(cudaMalloc(&fb.y0, fb.n * 4)); CHECK(cudaMalloc(&fb.y1, fb.n * 4));
	CHECK(cudaMemcpy(fb.base, fbase.data(), fb.n * 4, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&fb.logp, fb.n)); CHECK(cudaMalloc(&fb.R, fb.n * 4));
	CHECK(cudaMemcpy(fb.p, fp.data(), fb.n * 4, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(fb.y0, f0.data(), fb.n * 4, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(fb.y1, f1.data(), fb.n * 4, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(fb.logp, fl.data(), fb.n, cudaMemcpyHostToDevice));
	fprintf(stderr, "factor base: %d entries (%d below I = %u), set up in %.1f s\n", fb.n, nsmall, I,
		std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());

	// special-q list
	std::vector<QLattice> qs;
	if (qfile) {
		FILE* f = fopen(qfile, "r"); if (!f) { perror(qfile); return 1; }
		char line[1024]; int nbasis_diff = 0, nother = 0;
		while (fgets(line, sizeof line, f)) {
			unsigned long long q, r; long long a0, b0, a1, b1; unsigned int Ii, J; int sqside;
			if (sscanf(line, "# Sieving side-%d q=%llu; rho=%llu; a0=%lld; b0=%lld; a1=%lld; b1=%lld; I=%u; J=%u;",
			           &sqside, &q, &r, &a0, &b0, &a1, &b1, &Ii, &J) == 9) {
				if (sqside != side) { nother++; continue; }
				if (!qs.empty() && std::any_of(qs.begin(), qs.end(), [&](const QLattice& x) { return x.q == q && x.r == r; })) continue;   // las repeats a q after bucket resizing
				if (poly_norm) {
					uint64_t fq[MAXD + 1], v = 0; spoly_mod(q, fq);
					for (int k = sdeg; k >= 0; k--) v = (uint64_t)(((u128)v * r + fq[k]) % q);
					if (v != 0) { fprintf(stderr, "%s: rho=%llu is not a root modulo q=%llu of side %d\n", qfile, r, q, side); return 1; }
				}
				qs.push_back({q, r, a0, b0, a1, b1, J});
				QLattice M = reduce_q(q, poly_norm ? r : lin_root(q), P.skew, logI);   // compare with our own reduction
				bool same = M.r == r && M.J == J && ((M.a0 == a0 && M.b0 == b0) || (M.a0 == -a0 && M.b0 == -b0))
				            && ((M.a1 == a1 && M.b1 == b1) || (M.a1 == -a1 && M.b1 == -b1));
				if (!same) { nbasis_diff++; if (verbose) fprintf(stderr, "q=%llu: las basis (%lld,%lld) (%lld,%lld) J=%u, ours (%lld,%lld) (%lld,%lld) J=%u\n",
					q, a0, b0, a1, b1, J, (long long)M.a0, (long long)M.b0, (long long)M.a1, (long long)M.b1, M.J); }
			}
		}
		fclose(f);
		fprintf(stderr, "%zu special-q from %s; our own reduction differs from las for %d of them%s\n", qs.size(), qfile, nbasis_diff,
			nother ? " (special-q of the other side ignored)" : "");
	} else {
		std::vector<uint64_t> primes = primes_in(q0, q1);
		if (!poly_norm)       // q | c1: projective special-q, not handled
			primes.erase(std::remove_if(primes.begin(), primes.end(), [](uint64_t q) { return mod_str(g_spoly[1], q) == 0; }), primes.end());
		const bool sample = nrandom > 0 && nrandom < (long)primes.size();
		if (sample) { std::mt19937_64 g(1); std::shuffle(primes.begin(), primes.end(), g); }
		std::vector<std::pair<uint64_t, uint64_t>> qr;     // (q, root)
		if (!poly_norm) {
			if (sample) { primes.resize(nrandom); std::sort(primes.begin(), primes.end()); }
			for (uint64_t q : primes) qr.push_back({q, lin_root(q)});
		} else {              // all affine roots of each q (with -random: of random q until there are N special-q)
			for (size_t done = 0; done < primes.size() && (!sample || (long)qr.size() < nrandom); ) {
				const size_t n = sample ? std::min<size_t>(primes.size() - done, 4096) : primes.size() - done;
				std::vector<std::vector<uint64_t>> rr(n);
				parallel_for(n, nthreads, [&](size_t t) { rr[t] = spoly_roots(primes[done + t]); });
				for (size_t t = 0; t < n && (!sample || (long)qr.size() < nrandom); t++) for (uint64_t r : rr[t]) qr.push_back({primes[done + t], r});
				done += n;
			}
			std::sort(qr.begin(), qr.end());
		}
		qs.resize(qr.size());
		parallel_for(qr.size(), nthreads, [&](size_t t) { qs[t] = reduce_q(qr[t].first, qr[t].second, P.skew, logI); });
	}
	fprintf(stderr, "%zu special-q\n", qs.size());

	// chunks of large primes with equal expected hits (about I*J/p each), for k_buckets2
	const int nchunks = 320;
	std::vector<int> hchunk; {
		double total = 0; for (int k = nsmall; k < fb.n; k++) total += 1.0 / fp[k];
		double acc = 0; hchunk.push_back(nsmall);
		for (int k = nsmall; k < fb.n; k++) { acc += 1.0 / fp[k]; if (acc >= total * hchunk.size() / nchunks && (int)hchunk.size() < nchunks) hchunk.push_back(k + 1); }
		hchunk.push_back(fb.n);
	}
	int* dchunk; CHECK(cudaMalloc(&dchunk, hchunk.size() * sizeof(int)));
	CHECK(cudaMemcpy(dchunk, hchunk.data(), hchunk.size() * sizeof(int), cudaMemcpyHostToDevice));
	if (!oldbuckets) CHECK(cudaFuncSetAttribute(k_buckets2, cudaFuncAttributeMaxDynamicSharedMemorySize, (I >> 1) * 4));

	// Pipeline. Each lane has its own transformed roots, buckets and row counters; special-q k uses lane k % lanes. Transform
	// and buckets run on stream sb, the rows on stream sr, so with 2 lanes the buckets of special-q k+1 are filled while the
	// rows of special-q k are sieved (they use different units: global atomics in L2, shared-memory atomics). Two output
	// slots: the host copies and writes the survivors of special-q k (stream cs) while the GPU works on the next ones.
	// Bucket capacity per region: a large prime p > I hits a region of RS columns with probability RS/p, so the expected
	// number of entries is RS * sum(1/p); the counts are close to it (entries beyond cap are counted as dropped).
	// The rows of a special-q are sieved in nparts passes of Jmax/nparts rows (nparts a power of 2), each with the buckets of
	// its own rows only; the large primes' walks continue from one pass to the next (k_buckets, st: 8 bytes per large entry).
	const uint32_t Jmax = I >> 1;
	const int nlarge = fb.n - fb.nsmall;
	double inv_sum = 0; for (int k = nsmall; k < fb.n; k++) inv_sum += 1.0 / fp[k];
	const double expect = RS * inv_sum;
	const uint32_t cap = ((uint32_t)(1.3 * expect + 10 * sqrt(expect)) + 255) & ~255u;
	const unsigned int maxout = (unsigned int)std::min<double>(maxout_m * 1048576.0, 4e9);
	auto need_for = [&](uint32_t np) {
		const size_t nregp = (size_t)(Jmax / np) * rpr;
		return (size_t)lanes * nregp * ((size_t)cap + CPAD) * 4 + 2 * (size_t)maxout * sizeof(OutPair) + (size_t)lanes * fb.n * 8
		       + (np > 1 ? (size_t)lanes * nlarge * sizeof(uint2) : 0);
	};
	uint32_t nparts = 1;
	{
		size_t freeb = 0, totalb = 0; CHECK(cudaMemGetInfo(&freeb, &totalb));
		if (nparts_opt > 0) {
			if ((nparts_opt & (nparts_opt - 1)) != 0 || nparts_opt > (long)Jmax) {
				fprintf(stderr, "-parts must be a power of 2 up to %u (I/2)\n", Jmax); return 1;
			}
			nparts = (uint32_t)nparts_opt;
		} else {
			// as few passes as fit into the memory, and with at most L2/256 and at most 65536 bucket regions per pass, so that
			// the ends of the buckets being filled stay in the L2 cache. In tests, a 4.5 MB L2 was fastest with 16384 regions
			// per pass at I = 16 and 17 (1.5 times faster than one pass), a 96 MB L2 with 65536 at I = 16 to 19 (1.25 times
			// faster than 262144); more passes cost about 1 ms each.
			int l2 = 0; CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
			const size_t maxreg = std::min<size_t>((size_t)l2 / 256, 65536);
			while (nparts < Jmax && (need_for(nparts) > freeb * 0.95 || (size_t)(Jmax / nparts) * rpr > maxreg)) nparts *= 2;
		}
		const size_t need = need_for(nparts);
		fprintf(stderr, "buckets: %zu regions of %u columns (%u per row), %u entries each (expected %.0f), in %u pass%s of %u rows: "
			"%.2f GB on the GPU (%.2f GB free)\n", (size_t)Jmax * rpr, RS, rpr, cap, expect, nparts, nparts > 1 ? "es" : "",
			Jmax / nparts, need / 1e9, freeb / 1e9);
		if (need > freeb * 0.95) {
			fprintf(stderr, "not enough GPU memory%s\n", lanes > 1 ? ": try -lanes 1" : nparts_opt > 0 ? ": try a larger -parts" : "");
			return 1;
		}
	}
	if (!oldbuckets && nparts > 1) { fprintf(stderr, "-twopass needs a single pass (-parts 1)\n"); return 1; }
	const size_t nregp = (size_t)(Jmax / nparts) * rpr;      // bucket regions of one pass
	struct Lane { DevFB fb; uint32_t* bucket; uint32_t* count; uint2* st; cudaEvent_t rows_done; };
	std::vector<Lane> lane(lanes);
	for (Lane& L : lane) {
		L.fb = fb; CHECK(cudaMalloc(&L.fb.R, fb.n * 4));
		CHECK(cudaMalloc(&L.fb.proj, (size_t)std::max(fb.nsmall, 1) * 4)); CHECK(cudaMalloc(&L.fb.nproj, 4));
		CHECK(cudaMalloc(&L.fb.part, (size_t)std::max(fb.nsmall, 1) * 16)); CHECK(cudaMalloc(&L.fb.npart, 4));
		CHECK(cudaMalloc(&L.bucket, nregp * cap * 4)); CHECK(cudaMalloc(&L.count, nregp * CPAD * 4));
		L.st = nullptr;
		if (nparts > 1) CHECK(cudaMalloc(&L.st, (size_t)std::max(nlarge, 1) * sizeof(uint2)));
		CHECK(cudaEventCreateWithFlags(&L.rows_done, cudaEventDisableTiming));
	}
	struct Slot {
		OutPair* dout; unsigned int* dn; unsigned long long* dropped;          // device
		OutPair* hout; unsigned int* hn; unsigned long long* hdropped;         // pinned host
		cudaEvent_t e[6];                         // e[0..1]: transform timing; e[5]: n and dropped copied
		std::vector<cudaEvent_t> pe;              // per pass: buckets start, end, rows start, end
		uint32_t used;                            // passes of the special-q in this slot
	} slot[2];
	for (Slot& s : slot) {
		CHECK(cudaMalloc(&s.dout, (size_t)maxout * sizeof(OutPair))); CHECK(cudaMalloc(&s.dn, 4)); CHECK(cudaMalloc(&s.dropped, 8));
		CHECK(cudaMallocHost(&s.hout, (size_t)maxout * sizeof(OutPair))); CHECK(cudaMallocHost(&s.hn, 4));
		CHECK(cudaMallocHost(&s.hdropped, 8));
		for (cudaEvent_t& ev : s.e) CHECK(cudaEventCreate(&ev));
		s.pe.resize(4 * (size_t)nparts);
		for (cudaEvent_t& ev : s.pe) CHECK(cudaEventCreate(&ev));
		s.used = 0;
	}
	cudaStream_t sb, sr, cs;
	CHECK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking)); CHECK(cudaStreamCreateWithFlags(&sr, cudaStreamNonBlocking));
	CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
	int nsm = 0; CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
	FILE* out = fopen(outf, "wb");
	if (!out) { fprintf(stderr, "cannot write %s\n", outf); return 1; }
	// header: version 1, the side whose factors NFS_factory rebuilds (the other side; NFS_factory checks the sieved one)
	uint32_t header[4] = {0, 1, (uint32_t)(1 - side), 0}; memcpy(header, "NFAB", 4); fwrite(header, sizeof header, 1, out);
	std::vector<char> wbuf;
	// With a persistent row grid of rblocks blocks per SM, each block asks for 1/rblocks of the SM's shared memory, so that
	// exactly rblocks blocks run on every SM (otherwise the scheduler can stack them unevenly, and the SMs with more blocks
	// set the pace).
	size_t rsmem = RS;
	if (rblocks > 0) {
		int smem_sm = 0, smem_res = 0, smem_optin = 0;
		CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
		CHECK(cudaDeviceGetAttribute(&smem_res, cudaDevAttrReservedSharedMemoryPerBlock, 0));
		CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
		size_t share = (size_t)(smem_sm / rblocks - smem_res) & ~(size_t)255;
		rsmem = std::max<size_t>(RS, std::min<size_t>(share, smem_optin));
	}
	auto rows_kernel = poly_norm ? (rpr == 1 ? k_rows<ONE_REGION, true> : logI <= 16 ? k_rows<SPLIT32, true> : k_rows<SPLIT_WIDE, true>)
	                             : (rpr == 1 ? k_rows<ONE_REGION, false> : logI <= 16 ? k_rows<SPLIT32, false> : k_rows<SPLIT_WIDE, false>);
	CHECK(cudaFuncSetAttribute(rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)rsmem));
	const double Y0d = (double)strtold(g_spoly[0].c_str(), nullptr), Y1d = (double)strtold(g_spoly[1].c_str(), nullptr);
	std::vector<long double> sc; for (const std::string& s : g_spoly) sc.push_back(strtold(s.c_str(), nullptr));
	const int bgrid = bblocks > 0 ? nsm * bblocks : (nlarge + 127) / 128;
	fprintf(stderr, "%d SMs, %d lane%s, bucket grid %d x 128, row grid %s\n", nsm, lanes, lanes > 1 ? "s" : "", bgrid,
		rblocks > 0 ? (std::to_string(nsm * rblocks) + " x 512, " + std::to_string(rsmem) + " bytes shared each").c_str()
		            : "one block per region");

	auto enqueue = [&](size_t k) {
		const QLattice& Q = qs[k]; Lane& L = lane[k % lanes]; Slot& s = slot[k & 1];
		if (k >= (size_t)lanes) CHECK(cudaStreamWaitEvent(sb, L.rows_done, 0));   // rows of the lane's previous special-q done
		CHECK(cudaMemsetAsync(L.fb.nproj, 0, 4, sb)); CHECK(cudaMemsetAsync(L.fb.npart, 0, 4, sb));
		CHECK(cudaMemsetAsync(s.dropped, 0, 8, sb)); CHECK(cudaMemsetAsync(s.dn, 0, 4, sb));
		CHECK(cudaEventRecord(s.e[0], sb));
		k_transform<<<(fb.n + 255) / 256, 256, 0, sb>>>(L.fb, Q.q, Q.a0, Q.b0, Q.a1, Q.b1);
		CHECK(cudaEventRecord(s.e[1], sb));
		// value/q = alpha*i + beta*j with alpha = (Y1*a0 + Y0*b0)/q, beta = (Y1*a1 + Y0*b1)/q
		// (scaled by 2^-e: they can exceed the float range)
		float alpha = 0, beta = 0; int e;
		NormPoly np{}; np.d = sdeg; np.invS = 1.0f / (float)(I >> 1);
		if (!poly_norm) {
			double ad = (Y1d * (double)Q.a0 + Y0d * (double)Q.b0) / (double)Q.q, bd = (Y1d * (double)Q.a1 + Y0d * (double)Q.b1) / (double)Q.q;
			e = (int)floor(log2(std::max(fabs(ad), fabs(bd))));
			alpha = (float)ldexp(ad, -e); beta = (float)ldexp(bd, -e);
		} else {
			// F(i,j)/q = sum g[k] i^k j^(d-k) = S^d sum g[k] t^k u^(d-k) with S = I/2: G[k] = g[k] S^d 2^-e, max |G[k]| in [1, 2)
			std::vector<long double> g = homography<long double>(sc, Q.a0, Q.b0, Q.a1, Q.b1);
			long double mx = 0;
			for (long double& x : g) { x = ldexpl(x / (long double)Q.q, (logI - 1) * sdeg); mx = std::max(mx, fabsl(x)); }
			e = (int)floorl(log2l(mx));
			for (int k = 0; k <= sdeg; k++) np.G[k] = (float)ldexpl(g[k], -e);
		}
		// the passes over the rows: buckets of rows jlo .. jlo+jn-1 (stream sb), then those rows (stream sr); the next pass
		// refills the buckets only after the rows have read them
		const uint32_t Jp = Jmax / nparts;
		s.used = 0;
		for (uint32_t jlo = 0; jlo < Q.J; jlo += Jp) {
			const uint32_t jn = std::min(Jp, Q.J - jlo), t = s.used++;
			if (t > 0) CHECK(cudaStreamWaitEvent(sb, L.rows_done, 0));
			CHECK(cudaMemsetAsync(L.count, 0, (size_t)jn * rpr * CPAD * 4, sb));
			CHECK(cudaEventRecord(s.pe[4 * t], sb));
			if (oldbuckets) k_buckets<<<bgrid, 128, 0, sb>>>(L.fb, logI, logR, jlo, jlo + jn, L.st, t > 0, L.bucket, L.count, cap, s.dropped);
			else k_buckets2<<<(int)hchunk.size() - 1, 1024, Q.J * 4, sb>>>(L.fb, logI, Q.J, dchunk, L.bucket, L.count, cap, s.dropped);
			CHECK(cudaEventRecord(s.pe[4 * t + 1], sb));
			CHECK(cudaStreamWaitEvent(sr, s.pe[4 * t + 1], 0));
			CHECK(cudaEventRecord(s.pe[4 * t + 2], sr));
			const unsigned int nr = jn * rpr - (jlo ? 0 : rpr), rgrid = rblocks > 0 ? std::min<unsigned int>(nsm * rblocks, nr) : nr;
			if (nr > 0)
				rows_kernel<<<rgrid, 512, rsmem, sr>>>(L.fb, logI, logR, jlo, jn, L.bucket, L.count, cap, Q.a0, Q.b0, Q.a1, Q.b1, alpha, beta,
				                              np, (float)e, (float)(mfb + slack), dummy, dc1, dc0, (float)dummy_bits, s.dout, s.dn, maxout, tiny);
			CHECK(cudaGetLastError());
			CHECK(cudaEventRecord(s.pe[4 * t + 3], sr));
			CHECK(cudaEventRecord(L.rows_done, sr));
		}
		CHECK(cudaMemcpyAsync(s.hn, s.dn, 4, cudaMemcpyDeviceToHost, sr));
		CHECK(cudaMemcpyAsync(s.hdropped, s.dropped, 8, cudaMemcpyDeviceToHost, sr));
		CHECK(cudaEventRecord(s.e[5], sr));
	};

	double t_tr = 0, t_bk = 0, t_rows = 0, t_wait = 0; unsigned long long total_out = 0, total_drop = 0, positions = 0, total_trunc = 0;
	unsigned long long total_bigb = 0;     // survivors with b >= 2^32, which the .ab format cannot hold
	auto process = [&](size_t k) {
		const QLattice& Q = qs[k]; Slot& s = slot[k & 1];
		auto tw = std::chrono::steady_clock::now();
		CHECK(cudaEventSynchronize(s.e[5]));
		t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
		float ms1, ms2 = 0, ms3 = 0, m;
		cudaEventElapsedTime(&ms1, s.e[0], s.e[1]);
		for (uint32_t t = 0; t < s.used; t++) {
			cudaEventElapsedTime(&m, s.pe[4 * t], s.pe[4 * t + 1]); ms2 += m;
			cudaEventElapsedTime(&m, s.pe[4 * t + 2], s.pe[4 * t + 3]); ms3 += m;
		}
		t_tr += ms1; t_bk += ms2; t_rows += ms3;
		unsigned int n = *s.hn; unsigned long long dr = *s.hdropped;
		if (n > maxout) {
			fprintf(stderr, "q=%llu: %u survivors, only %u kept (raise -maxout)\n", (unsigned long long)Q.q, n, maxout);
			n = maxout; total_trunc++;
		}
		CHECK(cudaMemcpyAsync(s.hout, s.dout, (size_t)n * sizeof(OutPair), cudaMemcpyDeviceToHost, cs));
		CHECK(cudaStreamSynchronize(cs));
		// gcd(i,j) = 1 was checked on the GPU, so gcd(a,b) is 1 or q
		const int64_t q = (int64_t)Q.q;
		wbuf.resize((size_t)(n + 1) * 12); char* w = wbuf.data();
		for (unsigned int t = 0; t < n; t++) {
			int64_t a = s.hout[t].a, b = s.hout[t].b;
			if (b >= (1ll << 32)) { total_bigb++; continue; }
			if (a % q == 0 && b % q == 0) continue;
			uint32_t b32 = (uint32_t)b; memcpy(w, &a, 8); memcpy(w + 8, &b32, 4); w += 12;
		}
		{   // row j = 0 is not sieved; its only coprime point (i,j) = (1,0) is (a0,b0), a short vector with a small value
			int64_t a = Q.a0, b = Q.b0;
			if (b < 0) { a = -a; b = -b; }
			int64_t g = a < 0 ? -a : a, y = b;
			while (y) { int64_t t = g % y; g = y; y = t; }
			const bool dummy_ok = !dummy || log2(fabs((double)dc1 * (double)a + (double)dc0 * (double)b) + 1) < dummy_bits;
			if (b > 0 && b < (1ll << 32) && g == 1 && dummy_ok) {
				uint32_t b32 = (uint32_t)b; memcpy(w, &a, 8); memcpy(w + 8, &b32, 4); w += 12; n++;
			}
		}
		fwrite(wbuf.data(), 1, w - wbuf.data(), out);
		total_out += n; total_drop += dr; positions += (uint64_t)I * (Q.J - 1);
		if (verbose) fprintf(stderr, "q=%llu J=%u: %u survivors, %llu bucket entries dropped, %u pass%s, %.2f + %.2f + %.2f ms\n",
			(unsigned long long)Q.q, Q.J, n, dr, s.used, s.used > 1 ? "es" : "", ms1, ms2, ms3);
	};

	// the host stays up to two special-q ahead of the one it writes (output slot (k+2)&1 = k&1 is free only after process(k))
	// progress: a line every progress_s seconds (the last special-q written, the share done, the time left)
	auto t_loop = std::chrono::steady_clock::now();
	double next_report = progress_s;
	if (!qs.empty()) enqueue(0);
	for (size_t k = 0; k < qs.size(); k++) {
		if (k + 1 < qs.size()) enqueue(k + 1);
		process(k);
		const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count();
		if (progress_s > 0 && el >= next_report && k + 1 < qs.size()) {
			next_report = el + progress_s;
			const double left = el / (k + 1) * (qs.size() - k - 1);
			fprintf(stderr, "q=%llu: %zu of %zu special-q (%.1f %%), %.1f ms per q, %llu survivors (%.1f GB), %dh%02dm left\n",
				(unsigned long long)qs[k].q, k + 1, qs.size(), 100.0 * (k + 1) / qs.size(), 1000 * el / (k + 1), total_out,
				12e-9 * total_out, (int)(left / 3600), (int)fmod(left / 60, 60));
		}
	}
	fclose(out);
	double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_loop).count();
	fprintf(stderr, "%zu special-q, %.3g positions, %llu survivors (%llu bucket entries dropped) in %.2f s: %.1f ms per q "
		"(transform %.2f, buckets %.2f, rows %.2f ms; host waited %.1f ms), %.3g positions/s\n", qs.size(), (double)positions,
		total_out, total_drop, sec, 1000 * sec / qs.size(), t_tr / qs.size(), t_bk / qs.size(), t_rows / qs.size(),
		1000 * t_wait / qs.size(), positions / sec);
	if (total_bigb) fprintf(stderr, "WARNING: %llu survivors with b >= 2^32 were left out (b has 32 bits in .ab files): use smaller special-q or I\n",
		total_bigb);
	if (total_trunc) fprintf(stderr, "WARNING: %llu special-q had more than %u survivors (-maxout %g); the rest of their survivors were lost\n",
		total_trunc, maxout, maxout_m);
	return 0;
}
