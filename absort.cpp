// absort: sorts the pairs of .ab files and of text files of "a,b" lines numerically by (a, b) (a first, then b, both
// ascending), removes duplicates and writes one .ab file (NFS_factory's format of only a and b, as written by gpusieve,
// RelationsToBinaryFormat and absort itself). Sorted this way, an .ab file compresses best with xz:
//
//     xz -T0 -k --block-size=128MiB --delta=dist=12 --lzma2=preset=6,lc=1,lp=2,pb=2 store.ab
//
// (xz -d or pixz -d return the .ab unchanged.)
//
//     absort [-side 0|1] [-threads N] [-mem GB] [-tmp DIR] -o out.ab in1 [in2 ...]
//     absort -merge [-side 0|1] -o out.ab sorted1.ab [sorted2.ab ...]
//
// Inputs: .ab files (recognized by their header) and text files whose lines start with "a,b": NFS_factory's
// "format: keys", ab2keys output, or relations "a,b:..." (the rest of a line after b is ignored). Lines that do not start
// with a pair are counted and skipped. Inputs must be regular files (they are mapped into memory). -o - writes to
// standard output. The output may be one of the inputs: all inputs are read before it is written.
// -side: the side in the header of the output (0 rational, 1 algebraic). Default: the side of the .ab inputs (they must
// agree), or 0 if all inputs are text.
// Memory: 16 bytes per pair read (two arrays of 64-bit keys), plus the page cache of the inputs. absort uses at most
// 90% of the memory available when it starts (MemAvailable in /proc/meminfo), or -mem GB. If the pairs need more, it
// sorts them in batches that fit, writes the sorted unique keys of each batch (8 bytes per pair) to a temporary file in
// -tmp DIR (default: the folder of the output file, or the current folder for -o -), and then merges these files into
// the output in pieces of key ranges that fit. The temporary files are deleted as soon as they are created (absort
// keeps them open), so nothing is left behind if it stops, and the merge gives their space back as it goes.
// -merge: the inputs are .ab files that are already sorted, as absort writes them (- reads one of them from standard
// input). They are merged in one pass with a buffer of 3 MB each, duplicates removed, for example to add a newly sorted
// batch of pairs to an existing store without sorting the store again. An input that is not sorted stops the merge with
// an error. The output must not be one of the inputs.
//
// Build: g++ -O3 -fopenmp -o absort absort.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/falloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

namespace {

const size_t AB_HEADER = 16;
const size_t AB_RECORD = 12;
const int BUCKET_BITS = 16;
const size_t NB = (size_t)1 << BUCKET_BITS;
const size_t WRITE_PIECE = (size_t)1 << 18;   // records per write buffer of a thread

// Pairs with -2^31 <= a < 2^31 are kept as one 64-bit key whose unsigned order is the order of (a, b).
inline uint64_t make_key(int64_t a, uint32_t b) { return ((uint64_t)(uint32_t)(a + 0x80000000LL) << 32) | b; }
inline int64_t key_a(uint64_t k) { return (int64_t)(k >> 32) - 0x80000000LL; }
inline uint32_t key_b(uint64_t k) { return (uint32_t)k; }
inline bool fits_key(int64_t a) { return a >= -0x80000000LL && a < 0x80000000LL; }

// Pairs with a outside that range (none in practice) are kept apart, in memory.
struct Wide
{
	int64_t a;
	uint32_t b;
	bool operator<(const Wide& o) const { return a != o.a ? a < o.a : b < o.b; }
	bool operator==(const Wide& o) const { return a == o.a && b == o.b; }
};

struct Input
{
	std::string name;
	const char* data = nullptr;
	size_t size = 0;
	bool ab = false;
	uint32_t side = 0;
	unsigned long long pairs = 0, bad = 0;
	size_t last_chunk = 0;
};

struct Chunk
{
	int input;
	size_t begin, end;      // bytes of the file (text) or records (.ab)
	size_t bound;           // most pairs it can hold: its lines or records
	size_t offset;          // where its keys go in the batch
	size_t count;           // keys it actually held
};

// Sorted keys in buckets of their top bits: bucket b holds size[b] unique keys from sorted[start[b]], which go to
// position ostart[b] of the output.
struct Buckets
{
	std::vector<size_t> start, size, ostart;
	size_t unique = 0;
};

// A temporary file of sorted unique keys; pos: the keys before it are merged.
struct Run
{
	int fd;
	size_t n, pos = 0;
	off_t punched = 0;
};

double now()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t mem_available()
{
	FILE* f = fopen("/proc/meminfo", "r");
	if (!f)
	{
		return 0;
	}
	char line[256];
	unsigned long long kb = 0;
	while (fgets(line, sizeof line, f))
	{
		if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
		{
			break;
		}
	}
	fclose(f);
	return (size_t)kb * 1024;
}

size_t count_lines(const char* p, size_t len)
{
	size_t c = 0;
	for (size_t i = 0; i < len; i++)
	{
		c += (p[i] == '\n');
	}
	if (len && p[len - 1] != '\n')
	{
		c++;
	}
	return c;
}

// Parses the lines of [p, end): keys to out, pairs with a large a to wide. Returns the number of keys.
size_t parse_text(const char* p, const char* end, uint64_t* out, std::vector<Wide>& wide, unsigned long long& pairs,
	unsigned long long& bad)
{
	size_t n = 0;
	while (p < end)
	{
		const char* nl = (const char*)memchr(p, '\n', (size_t)(end - p));
		const char* le = nl ? nl : end;
		const char* q = p;
		bool neg = false;
		if (q < le && *q == '-')
		{
			neg = true;
			q++;
		}
		uint64_t a = 0, b = 0;
		int da = 0, db = 0;
		while (q < le && *q >= '0' && *q <= '9' && da < 18)
		{
			a = a * 10 + (uint64_t)(*q - '0');
			q++;
			da++;
		}
		bool ok = da > 0 && q < le && *q == ',';
		if (ok)
		{
			q++;
			while (q < le && *q >= '0' && *q <= '9' && db < 11)
			{
				b = b * 10 + (uint64_t)(*q - '0');
				q++;
				db++;
			}
			ok = db > 0 && b > 0 && b <= 0xFFFFFFFFull && (q == le || *q == ':' || *q == '\r' || *q == ' ' || *q == '\t');
		}
		if (ok)
		{
			int64_t av = neg ? -(int64_t)a : (int64_t)a;
			if (fits_key(av))
			{
				out[n++] = make_key(av, (uint32_t)b);
			}
			else
			{
				wide.push_back({ av, (uint32_t)b });
			}
			pairs++;
		}
		else if (le > p && !(le == p + 1 && *p == '\r'))
		{
			bad++;
		}
		p = nl ? nl + 1 : end;
	}
	return n;
}

size_t parse_ab(const char* rec, size_t count, uint64_t* out, std::vector<Wide>& wide)
{
	size_t n = 0;
	for (size_t i = 0; i < count; i++, rec += AB_RECORD)
	{
		int64_t a;
		uint32_t b;
		memcpy(&a, rec, 8);
		memcpy(&b, rec + 8, 4);
		if (fits_key(a))
		{
			out[n++] = make_key(a, b);
		}
		else
		{
			wide.push_back({ a, b });
		}
	}
	return n;
}

void put_record(char* p, int64_t a, uint32_t b)
{
	memcpy(p, &a, 8);
	memcpy(p + 8, &b, 4);
}

bool write_all(int fd, const char* p, size_t len, off_t at, bool positioned)
{
	while (len)
	{
		ssize_t w = positioned ? pwrite(fd, p, len, at) : write(fd, p, len);
		if (w <= 0)
		{
			return false;
		}
		p += w;
		len -= (size_t)w;
		at += w;
	}
	return true;
}

bool read_all(int fd, char* p, size_t len, off_t at)
{
	while (len)
	{
		ssize_t r = pread(fd, p, len, at);
		if (r <= 0)
		{
			return false;
		}
		p += r;
		len -= (size_t)r;
		at += r;
	}
	return true;
}

// Buckets keys[0, n) by their top bits into sorted[], then sorts and deduplicates each bucket in place.
void sort_unique(const uint64_t* keys, size_t n, uint64_t* sorted, int threads, Buckets& bk)
{
	uint64_t kmin = UINT64_MAX, kmax = 0;
#pragma omp parallel for reduction(min : kmin) reduction(max : kmax)
	for (size_t i = 0; i < n; i++)
	{
		kmin = std::min(kmin, keys[i]);
		kmax = std::max(kmax, keys[i]);
	}
	int shift = 0;
	if (n)
	{
		while (((kmax - kmin) >> shift) >= NB)
		{
			shift++;
		}
	}
	std::vector<size_t> hist((size_t)threads * NB, 0);
	bk.start.assign(NB + 1, 0);
	bk.size.assign(NB, 0);
	bk.ostart.assign(NB + 1, 0);
#pragma omp parallel num_threads(threads)
	{
		const int t = omp_get_thread_num(), nt = omp_get_num_threads();
		const size_t lo = n * (size_t)t / (size_t)nt, hi = n * (size_t)(t + 1) / (size_t)nt;
		size_t* h = &hist[(size_t)t * NB];
		for (size_t i = lo; i < hi; i++)
		{
			h[(keys[i] - kmin) >> shift]++;
		}
#pragma omp barrier
#pragma omp single
		{
			size_t pos = 0;
			for (size_t b = 0; b < NB; b++)
			{
				bk.start[b] = pos;
				for (int u = 0; u < nt; u++)
				{
					size_t c = hist[(size_t)u * NB + b];
					hist[(size_t)u * NB + b] = pos;
					pos += c;
				}
			}
			bk.start[NB] = pos;
		}
		for (size_t i = lo; i < hi; i++)
		{
			sorted[h[(keys[i] - kmin) >> shift]++] = keys[i];
		}
	}
#pragma omp parallel for schedule(dynamic, 16)
	for (size_t b = 0; b < NB; b++)
	{
		uint64_t* s = sorted + bk.start[b];
		uint64_t* e = sorted + bk.start[b + 1];
		std::sort(s, e);
		bk.size[b] = (size_t)(std::unique(s, e) - s);
	}
	for (size_t b = 0; b < NB; b++)
	{
		bk.ostart[b + 1] = bk.ostart[b] + bk.size[b];
	}
	bk.unique = bk.ostart[NB];
}

// Writes the unique keys of the buckets as .ab records, from byte at of the file. A regular file is written in
// parallel, each bucket at its place; anything else in order.
bool write_records(int fd, bool positioned, off_t at, const uint64_t* sorted, const Buckets& bk)
{
	if (positioned)
	{
		bool wok = true;
#pragma omp parallel reduction(&& : wok)
		{
			std::vector<char> buf(WRITE_PIECE * AB_RECORD);
#pragma omp for schedule(dynamic, 16)
			for (size_t b = 0; b < NB; b++)
			{
				const uint64_t* s = sorted + bk.start[b];
				for (size_t i = 0; i < bk.size[b]; i += WRITE_PIECE)
				{
					size_t m = std::min(WRITE_PIECE, bk.size[b] - i);
					for (size_t j = 0; j < m; j++)
					{
						put_record(&buf[j * AB_RECORD], key_a(s[i + j]), key_b(s[i + j]));
					}
					wok = write_all(fd, buf.data(), m * AB_RECORD, at + (off_t)((bk.ostart[b] + i) * AB_RECORD), true) && wok;
				}
			}
		}
		return wok;
	}
	std::vector<char> buf(WRITE_PIECE * AB_RECORD);
	size_t m = 0;
	for (size_t b = 0; b < NB; b++)
	{
		const uint64_t* s = sorted + bk.start[b];
		for (size_t i = 0; i < bk.size[b]; i++)
		{
			put_record(&buf[m * AB_RECORD], key_a(s[i]), key_b(s[i]));
			if (++m == WRITE_PIECE)
			{
				if (!write_all(fd, buf.data(), m * AB_RECORD, 0, false))
				{
					return false;
				}
				m = 0;
			}
		}
	}
	return write_all(fd, buf.data(), m * AB_RECORD, 0, false);
}

// Writes the unique keys of the buckets as 64-bit keys to a temporary file, in parallel.
bool write_keys(int fd, const uint64_t* sorted, const Buckets& bk)
{
	bool wok = true;
#pragma omp parallel for schedule(dynamic, 16) reduction(&& : wok)
	for (size_t b = 0; b < NB; b++)
	{
		if (bk.size[b])
		{
			wok = write_all(fd, (const char*)(sorted + bk.start[b]), bk.size[b] * sizeof(uint64_t),
				(off_t)(bk.ostart[b] * sizeof(uint64_t)), true) && wok;
		}
	}
	return wok;
}

// First position >= r.pos of a key >= x in the run.
size_t run_lower_bound(const Run& r, uint64_t x, bool& ok)
{
	size_t lo = r.pos, hi = r.n;
	while (lo < hi)
	{
		size_t mid = lo + (hi - lo) / 2;
		uint64_t v;
		if (pread(r.fd, &v, sizeof v, (off_t)(mid * sizeof v)) != (ssize_t)sizeof v)
		{
			ok = false;
			return lo;
		}
		if (v < x)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}
	return lo;
}

std::string folder_of(const std::string& path)
{
	size_t s = path.rfind('/');
	if (s == std::string::npos)
	{
		return ".";
	}
	return s == 0 ? "/" : path.substr(0, s);
}

// -merge: an input .ab file already sorted by (a, b), read through a buffer of its own.
struct MergeInput
{
	std::string name;
	FILE* f = nullptr;
	std::vector<char> buf;
	size_t pos = 0, len = 0, tail = 0;   // tail: bytes of an incomplete pair at its end
	int64_t a = 0;
	uint32_t b = 0;
	unsigned long long pairs = 0;
};

// The next pair of the input into (a, b); false at its end.
bool merge_read(MergeInput& m)
{
	if (m.len - m.pos < AB_RECORD)
	{
		const size_t rest = m.len - m.pos;
		memmove(m.buf.data(), m.buf.data() + m.pos, rest);
		m.len = rest + fread(m.buf.data() + rest, 1, m.buf.size() - rest, m.f);
		m.pos = 0;
		if (m.len < AB_RECORD)
		{
			m.tail = m.len;
			return false;
		}
	}
	memcpy(&m.a, m.buf.data() + m.pos, 8);
	memcpy(&m.b, m.buf.data() + m.pos + 8, 4);
	m.pos += AB_RECORD;
	m.pairs++;
	return true;
}

bool same_file(const std::string& x, const std::string& y)
{
	struct stat s, t;
	return x != "-" && y != "-" && stat(x.c_str(), &s) == 0 && stat(y.c_str(), &t) == 0 && s.st_dev == t.st_dev && s.st_ino == t.st_ino;
}

// Merges .ab files that are each sorted by (a, b), as absort writes them, into one, in one pass and with a buffer of 3 MB
// per input, and removes the duplicates. An input that is not sorted stops the merge with an error.
int merge_sorted(const std::vector<Input>& inputs, int side, const std::string& outname)
{
	const double t0 = now();
	std::vector<MergeInput> in(inputs.size());
	int in_side = -1;
	for (size_t k = 0; k < inputs.size(); k++)
	{
		MergeInput& m = in[k];
		m.name = inputs[k].name;
		if (same_file(m.name, outname))
		{
			fprintf(stderr, "absort: %s is an input and the output (with -merge, write to another file)\n", m.name.c_str());
			return 1;
		}
		m.f = m.name == "-" ? stdin : fopen(m.name.c_str(), "rb");
		unsigned char h[AB_HEADER];
		uint32_t version = 0, s = 0;
		if (!m.f || fread(h, 1, AB_HEADER, m.f) != AB_HEADER || memcmp(h, "NFAB", 4) != 0)
		{
			fprintf(stderr, "absort: %s is not an .ab file (-merge reads only .ab files)\n", m.name.c_str());
			return 1;
		}
		memcpy(&version, h + 4, 4);
		memcpy(&s, h + 8, 4);
		if (version != 1)
		{
			fprintf(stderr, "absort: %s is an .ab file of version %u, not 1\n", m.name.c_str(), version);
			return 1;
		}
		if (in_side < 0)
		{
			in_side = (int)s;
		}
		else if (in_side != (int)s && side < 0)
		{
			fprintf(stderr, "absort: the .ab inputs are of different sides (%s: %u); choose one with -side\n", m.name.c_str(), s);
			return 1;
		}
		m.buf.resize(AB_RECORD << 18);
	}
	if (side < 0)
	{
		side = in_side;
	}
	FILE* out = outname == "-" ? stdout : fopen(outname.c_str(), "wb");
	if (!out)
	{
		fprintf(stderr, "absort: cannot create %s\n", outname.c_str());
		return 1;
	}
	char header[AB_HEADER];
	const uint32_t hv[3] = { 1, (uint32_t)side, 0 };
	memcpy(header, "NFAB", 4);
	memcpy(header + 4, hv, 12);
	bool ok = fwrite(header, 1, AB_HEADER, out) == AB_HEADER;

	// A binary heap of the inputs by their current pair, kept in the heap as (a with its sign bit flipped, b), whose unsigned
	// order is the order of (a, b). The smallest is written (unless it equals the pair written last), replaced by the next
	// pair of its input, and sifted down.
	struct Node
	{
		uint64_t hi;
		uint32_t lo, input;
		bool operator<(const Node& o) const { return hi != o.hi ? hi < o.hi : lo < o.lo; }
	};
	auto node = [&](uint32_t k) { return Node{ (uint64_t)in[k].a ^ (1ull << 63), in[k].b, k }; };
	std::vector<Node> heap;
	for (uint32_t k = 0; k < (uint32_t)in.size(); k++)
	{
		if (merge_read(in[k]))
		{
			heap.push_back(node(k));
		}
	}
	auto sift_down = [&](size_t i)
	{
		const size_t n = heap.size();
		const Node x = heap[i];
		for (;;)
		{
			size_t c = 2 * i + 1;
			if (c >= n)
			{
				break;
			}
			if (c + 1 < n && heap[c + 1] < heap[c])
			{
				c++;
			}
			if (!(heap[c] < x))
			{
				break;
			}
			heap[i] = heap[c];
			i = c;
		}
		heap[i] = x;
	};
	for (size_t i = heap.size() / 2; i-- > 0;)
	{
		sift_down(i);
	}
	std::vector<char> obuf(AB_RECORD << 20);
	size_t used = 0;
	unsigned long long written = 0;
	bool have_last = false;
	Node last{ 0, 0, 0 };
	auto fail = [&](void)
	{
		if (out != stdout)
		{
			fclose(out);
			unlink(outname.c_str());
		}
		return 1;
	};
	while (!heap.empty() && ok)
	{
		const Node top = heap[0];
		if (!have_last || top.hi != last.hi || top.lo != last.lo)
		{
			if (used == obuf.size())
			{
				ok = fwrite(obuf.data(), 1, used, out) == used;
				used = 0;
			}
			put_record(&obuf[used], (int64_t)(top.hi ^ (1ull << 63)), top.lo);
			used += AB_RECORD;
			written++;
			last = top;
			have_last = true;
		}
		MergeInput& m = in[top.input];
		if (merge_read(m))
		{
			heap[0] = node(top.input);
			if (heap[0] < top)
			{
				fprintf(stderr, "absort: %s is not sorted by (a, b) (pair %llu: %lld,%u after %lld,%u); sort it with absort first\n",
					m.name.c_str(), m.pairs, (long long)m.a, m.b, (long long)(top.hi ^ (1ull << 63)), top.lo);
				return fail();
			}
		}
		else
		{
			heap[0] = heap.back();
			heap.pop_back();
		}
		if (!heap.empty())
		{
			sift_down(0);
		}
	}
	ok = ok && fwrite(obuf.data(), 1, used, out) == used;
	if (out != stdout)
	{
		ok = fclose(out) == 0 && ok;
	}
	else
	{
		ok = fflush(out) == 0 && ok;
	}
	if (!ok)
	{
		fprintf(stderr, "absort: writing %s failed\n", outname.c_str());
		return 1;
	}
	unsigned long long total = 0;
	for (MergeInput& m : in)
	{
		if (m.tail)
		{
			fprintf(stderr, "absort: warning: %s ends with %zu bytes of an incomplete pair (ignored)\n", m.name.c_str(), m.tail);
		}
		if (ferror(m.f))
		{
			fprintf(stderr, "absort: reading %s failed\n", m.name.c_str());
			return 1;
		}
		fprintf(stderr, "absort: %s: %llu pairs\n", m.name.c_str(), m.pairs);
		total += m.pairs;
		if (m.f != stdin)
		{
			fclose(m.f);
		}
	}
	fprintf(stderr, "absort: merged %llu pairs of %zu files; %llu duplicates removed, wrote %llu pairs (side %d) to %s in %.1f s\n",
		total, in.size(), total - written, written, side, outname.c_str(), now() - t0);
	return 0;
}

int usage()
{
	fprintf(stderr, "usage: absort [-side 0|1] [-threads N] [-mem GB] [-tmp DIR] -o out.ab in1 [in2 ...]\n"
		"       absort -merge [-side 0|1] -o out.ab sorted1.ab [sorted2.ab ...]\n"
		"  sorts the pairs of .ab and text (\"a,b\" lines) files by (a, b), removes duplicates, writes one .ab file;\n"
		"  uses at most 90%% of the available memory (or -mem GB), and temporary files in DIR if it needs more.\n"
		"  -merge: merges .ab files that are already sorted (absort output) in one pass, with little memory\n");
	return 1;
}

} // namespace

int main(int argc, char** argv)
{
	int side = -1, threads = omp_get_max_threads();
	double mem_gb = 0;
	bool merge = false;
	std::string outname, tmpdir;
	std::vector<Input> inputs;
	for (int i = 1; i < argc; i++)
	{
		std::string s = argv[i];
		if (s == "-merge")
		{
			merge = true;
		}
		else if (s == "-side" && i + 1 < argc)
		{
			side = atoi(argv[++i]);
			if (side != 0 && side != 1)
			{
				return usage();
			}
		}
		else if (s == "-threads" && i + 1 < argc)
		{
			threads = atoi(argv[++i]);
		}
		else if (s == "-mem" && i + 1 < argc)
		{
			mem_gb = atof(argv[++i]);
			if (mem_gb <= 0)
			{
				return usage();
			}
		}
		else if (s == "-tmp" && i + 1 < argc)
		{
			tmpdir = argv[++i];
		}
		else if (s == "-o" && i + 1 < argc)
		{
			outname = argv[++i];
		}
		else if (s.size() > 1 && s[0] == '-')
		{
			return usage();
		}
		else
		{
			Input in;
			in.name = s;
			inputs.push_back(in);
		}
	}
	if (outname.empty() || inputs.empty() || threads < 1)
	{
		return usage();
	}
	if (merge)
	{
		return merge_sorted(inputs, side, outname);
	}
	omp_set_num_threads(threads);
	const bool to_stdout = outname == "-";
	if (tmpdir.empty())
	{
		tmpdir = to_stdout ? "." : folder_of(outname);
	}
	double t0 = now();

	// Memory: the two key arrays get what the limit leaves after the per-thread buffers and histograms.
	const size_t avail = mem_available();
	const double limit = mem_gb > 0 ? mem_gb * 1e9 : avail ? 0.9 * (double)avail : 1e300;
	const double overhead = (double)threads * (double)(NB * sizeof(size_t) + WRITE_PIECE * AB_RECORD) + (double)(64 << 20);
	const size_t cap = (size_t)std::max<double>((double)((size_t)1 << 20),
		std::min<double>(limit > overhead ? (limit - overhead) / 16 : 0, (double)((size_t)1 << 62)));

	// Map the inputs and cut them into chunks.
	std::vector<Chunk> chunks;
	int ab_side = -1;
	for (int f = 0; f < (int)inputs.size(); f++)
	{
		Input& in = inputs[f];
		int fd = open(in.name.c_str(), O_RDONLY);
		struct stat st;
		if (fd < 0 || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode))
		{
			fprintf(stderr, "absort: cannot open %s as a regular file\n", in.name.c_str());
			return 1;
		}
		in.size = (size_t)st.st_size;
		if (in.size == 0)
		{
			close(fd);
			continue;
		}
		void* m = mmap(nullptr, in.size, PROT_READ, MAP_PRIVATE, fd, 0);
		close(fd);
		if (m == MAP_FAILED)
		{
			fprintf(stderr, "absort: cannot map %s\n", in.name.c_str());
			return 1;
		}
		in.data = (const char*)m;
		in.ab = in.size >= AB_HEADER && memcmp(in.data, "NFAB", 4) == 0;
		if (in.ab)
		{
			uint32_t version;
			memcpy(&version, in.data + 4, 4);
			memcpy(&in.side, in.data + 8, 4);
			if (version != 1)
			{
				fprintf(stderr, "absort: %s is an .ab file of version %u, not 1\n", in.name.c_str(), version);
				return 1;
			}
			if (ab_side < 0)
			{
				ab_side = (int)in.side;
			}
			else if (ab_side != (int)in.side && side < 0)
			{
				fprintf(stderr, "absort: the .ab inputs are of different sides (%s: %u); choose one with -side\n",
					in.name.c_str(), in.side);
				return 1;
			}
			size_t records = (in.size - AB_HEADER) / AB_RECORD;
			if ((in.size - AB_HEADER) % AB_RECORD)
			{
				fprintf(stderr, "absort: warning: %s ends with %zu bytes of an incomplete pair (ignored)\n", in.name.c_str(),
					(in.size - AB_HEADER) % AB_RECORD);
			}
			const size_t per = std::min<size_t>((size_t)8 << 20, cap);
			for (size_t r = 0; r < records; r += per)
			{
				chunks.push_back({ f, r, std::min(records, r + per), 0, 0, 0 });
			}
		}
		else
		{
			// A chunk of per bytes plus the rest of its last line has at most per + 1 lines.
			const size_t per = std::min<size_t>((size_t)128 << 20, cap - 1);
			size_t pos = 0;
			while (pos < in.size)
			{
				size_t end = pos + per;
				if (end >= in.size)
				{
					end = in.size;
				}
				else
				{
					const char* nl = (const char*)memchr(in.data + end, '\n', in.size - end);
					end = nl ? (size_t)(nl - in.data) + 1 : in.size;
				}
				chunks.push_back({ f, pos, end, 0, 0, 0 });
				pos = end;
			}
		}
		in.last_chunk = chunks.size();
	}
	if (side < 0)
	{
		side = ab_side >= 0 ? ab_side : 0;
	}

	// Bounds of the chunks (lines or records), and batches of chunks that fit into memory.
#pragma omp parallel for schedule(dynamic, 1)
	for (size_t c = 0; c < chunks.size(); c++)
	{
		Chunk& ch = chunks[c];
		const Input& in = inputs[ch.input];
		ch.bound = in.ab ? ch.end - ch.begin : count_lines(in.data + ch.begin, ch.end - ch.begin);
	}
	std::vector<std::pair<size_t, size_t>> batches;
	size_t bound = 0, first = 0, cur = 0, maxb = 0;
	for (size_t c = 0; c < chunks.size(); c++)
	{
		if (cur && cur + chunks[c].bound > cap)
		{
			batches.push_back({ first, c });
			maxb = std::max(maxb, cur);
			first = c;
			cur = 0;
		}
		chunks[c].offset = cur;
		cur += chunks[c].bound;
		bound += chunks[c].bound;
	}
	batches.push_back({ first, chunks.size() });
	maxb = std::max(maxb, cur);
	const bool external = batches.size() > 1;
	fprintf(stderr, "absort: up to %zu pairs in %zu input files, %d threads, %.1f GB of memory for the keys", bound,
		inputs.size(), threads, 16.0 * (double)maxb / 1e9);
	if (limit < 1e299)
	{
		fprintf(stderr, " (limit %.1f GB%s)", limit / 1e9, mem_gb > 0 ? "" : ", 90% of the available memory");
	}
	fprintf(stderr, "\n");
	if (external)
	{
		fprintf(stderr, "absort: sorting in %zu batches, with temporary files (up to %.1f GB) in %s\n", batches.size(),
			8.0 * (double)bound / 1e9, tmpdir.c_str());
		struct statvfs vs;
		if (statvfs(tmpdir.c_str(), &vs) == 0 && (double)vs.f_bavail * (double)vs.f_frsize < 8.0 * (double)bound)
		{
			fprintf(stderr, "absort: warning: %s has only %.1f GB free\n", tmpdir.c_str(),
				(double)vs.f_bavail * (double)vs.f_frsize / 1e9);
		}
	}

	uint64_t* keys = (uint64_t*)malloc(std::max<size_t>(maxb, 1) * sizeof(uint64_t));
	uint64_t* sorted = (uint64_t*)malloc(std::max<size_t>(maxb, 1) * sizeof(uint64_t));
	if (!keys || !sorted)
	{
		fprintf(stderr, "absort: cannot allocate %.1f GB\n", 16.0 * (double)maxb / 1e9);
		return 1;
	}

	// Read, sort and deduplicate each batch; with several batches, write each to a temporary file.
	std::vector<std::vector<Wide>> wide_t(threads);
	std::vector<unsigned long long> pairs_c(chunks.size()), bad_c(chunks.size());
	std::vector<Run> runs;
	Buckets bk;
	unsigned long long total = 0;
	double t_read = 0, t_sort = 0;
	for (size_t bi = 0; bi < batches.size(); bi++)
	{
		const size_t c0 = batches[bi].first, c1 = batches[bi].second;
		double tb = now();
#pragma omp parallel for schedule(dynamic, 1)
		for (size_t c = c0; c < c1; c++)
		{
			Chunk& ch = chunks[c];
			const Input& in = inputs[ch.input];
			std::vector<Wide>& wide = wide_t[omp_get_thread_num()];
			if (in.ab)
			{
				size_t before = wide.size();
				ch.count = parse_ab(in.data + AB_HEADER + ch.begin * AB_RECORD, ch.end - ch.begin, keys + ch.offset, wide);
				pairs_c[c] = ch.count + (wide.size() - before);
			}
			else
			{
				ch.count = parse_text(in.data + ch.begin, in.data + ch.end, keys + ch.offset, wide, pairs_c[c], bad_c[c]);
			}
		}
		size_t n = 0;
		unsigned long long batch_pairs = 0;
		for (size_t c = c0; c < c1; c++)
		{
			Chunk& ch = chunks[c];
			if (n != ch.offset)
			{
				memmove(keys + n, keys + ch.offset, ch.count * sizeof(uint64_t));
			}
			n += ch.count;
			inputs[ch.input].pairs += pairs_c[c];
			inputs[ch.input].bad += bad_c[c];
			batch_pairs += pairs_c[c];
		}
		total += batch_pairs;
		for (Input& in : inputs)
		{
			if (in.data && in.last_chunk <= c1)
			{
				munmap((void*)in.data, in.size);
				in.data = nullptr;
			}
		}
		double tr = now();
		sort_unique(keys, n, sorted, threads, bk);
		double ts = now();
		t_read += tr - tb;
		t_sort += ts - tr;
		if (external)
		{
			std::string name = tmpdir + "/absort." + std::to_string(getpid()) + "." + std::to_string(bi);
			int fd = open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
			if (fd < 0)
			{
				fprintf(stderr, "absort: cannot create the temporary file %s\n", name.c_str());
				return 1;
			}
			unlink(name.c_str());
			if (!write_keys(fd, sorted, bk))
			{
				fprintf(stderr, "absort: writing a temporary file in %s failed (disk full?)\n", tmpdir.c_str());
				return 1;
			}
			runs.push_back({ fd, bk.unique });
			fprintf(stderr, "absort: batch %zu of %zu: %llu pairs, %zu after removing duplicates; read %.1f s, sorted %.1f s, "
				"written %.1f s\n", bi + 1, batches.size(), batch_pairs, bk.unique, tr - tb, ts - tr, now() - ts);
		}
	}
	for (const Input& in : inputs)
	{
		fprintf(stderr, "absort: %s: %s, %llu pairs", in.name.c_str(), in.ab ? ".ab" : "text", in.pairs);
		if (in.bad)
		{
			fprintf(stderr, ", %llu lines without a pair skipped", in.bad);
		}
		fprintf(stderr, "\n");
	}
	std::vector<Wide> wide;
	for (auto& w : wide_t)
	{
		wide.insert(wide.end(), w.begin(), w.end());
		std::vector<Wide>().swap(w);
	}
	std::sort(wide.begin(), wide.end());
	wide.erase(std::unique(wide.begin(), wide.end()), wide.end());
	size_t wide_low = (size_t)(std::partition_point(wide.begin(), wide.end(), [](const Wide& w) { return w.a < 0; }) - wide.begin());
	if (!external)
	{
		fprintf(stderr, "absort: read %llu pairs in %.1f s, sorted in %.1f s; %llu duplicates removed, %zu pairs remain", total,
			t_read, t_sort, total - (unsigned long long)(bk.unique + wide.size()), bk.unique + wide.size());
		if (!wide.empty())
		{
			fprintf(stderr, " (%zu with |a| of 2^31 or more)", wide.size());
		}
		fprintf(stderr, "\n");
	}
	double t3 = now();

	// Write: header, pairs with a < -2^31, the keys in order, pairs with a >= 2^31.
	int fd = to_stdout ? 1 : open(outname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
	struct stat ost;
	if (fd < 0 || fstat(fd, &ost) != 0)
	{
		fprintf(stderr, "absort: cannot create %s\n", outname.c_str());
		return 1;
	}
	const bool positioned = S_ISREG(ost.st_mode);
	char header[AB_HEADER];
	const uint32_t hv[3] = { 1, (uint32_t)side, 0 };
	memcpy(header, "NFAB", 4);
	memcpy(header + 4, hv, 12);
	bool ok = write_all(fd, header, AB_HEADER, 0, positioned);
	off_t at = (off_t)AB_HEADER;
	std::vector<char> wbuf;
	auto write_wide = [&](size_t from, size_t to)
	{
		wbuf.resize((to - from) * AB_RECORD);
		for (size_t i = from; i < to; i++)
		{
			put_record(&wbuf[(i - from) * AB_RECORD], wide[i].a, wide[i].b);
		}
		ok = ok && write_all(fd, wbuf.data(), wbuf.size(), at, positioned);
		at += (off_t)wbuf.size();
	};
	write_wide(0, wide_low);
	const off_t keys_at = at;
	size_t unique_keys = 0;
	if (!external)
	{
		ok = ok && write_records(fd, positioned, keys_at, sorted, bk);
		unique_keys = bk.unique;
	}
	else
	{
		// Merge the temporary files in pieces of key ranges [prev, x) that fit into memory: a piece is read into keys,
		// sorted and deduplicated as a batch, and written after the pieces before it.
		size_t remaining = 0;
		for (const Run& r : runs)
		{
			remaining += r.n;
		}
		const unsigned long long merged_in = remaining;
		uint64_t prev = 0;
		bool rok = true;
		int piece = 0;
		std::vector<size_t> end(runs.size()), off(runs.size() + 1);
		while (remaining && ok && rok)
		{
			double tp = now();
			if (remaining <= maxb)
			{
				for (size_t r = 0; r < runs.size(); r++)
				{
					end[r] = runs[r].n;
				}
			}
			else
			{
				// The largest x with at most maxb keys in [prev, x): each run holds a key at most once, so a piece gets
				// at least maxb - (number of runs) + 1 keys.
				auto count_below = [&](uint64_t x)
				{
					size_t s = 0;
					for (const Run& r : runs)
					{
						s += run_lower_bound(r, x, rok) - r.pos;
					}
					return s;
				};
				uint64_t lo = prev, hi = UINT64_MAX, x;
				if (count_below(hi) <= maxb)
				{
					x = hi;
				}
				else
				{
					while (hi - lo > 1)
					{
						uint64_t mid = lo + (hi - lo) / 2;
						if (count_below(mid) <= maxb)
						{
							lo = mid;
						}
						else
						{
							hi = mid;
						}
					}
					x = lo;
				}
				for (size_t r = 0; r < runs.size(); r++)
				{
					end[r] = run_lower_bound(runs[r], x, rok);
				}
				prev = x;
			}
			off[0] = 0;
			for (size_t r = 0; r < runs.size(); r++)
			{
				off[r + 1] = off[r] + (end[r] - runs[r].pos);
			}
			const size_t m = off[runs.size()];
			// Read the segments in blocks of up to 4M keys, in parallel.
			std::vector<std::pair<size_t, size_t>> blocks;   // run, first key of the block
			const size_t blk = (size_t)4 << 20;
			for (size_t r = 0; r < runs.size(); r++)
			{
				for (size_t k = runs[r].pos; k < end[r]; k += blk)
				{
					blocks.push_back({ r, k });
				}
			}
#pragma omp parallel for schedule(dynamic, 1) reduction(&& : rok)
			for (size_t i = 0; i < blocks.size(); i++)
			{
				const size_t r = blocks[i].first, k = blocks[i].second, len = std::min(blk, end[r] - k);
				rok = read_all(runs[r].fd, (char*)(keys + off[r] + (k - runs[r].pos)), len * sizeof(uint64_t),
					(off_t)(k * sizeof(uint64_t))) && rok;
			}
			sort_unique(keys, m, sorted, threads, bk);
			ok = ok && write_records(fd, positioned, at, sorted, bk);
			at += (off_t)(bk.unique * AB_RECORD);
			unique_keys += bk.unique;
			remaining -= m;
			for (size_t r = 0; r < runs.size(); r++)
			{
				// Give back the disk space of the merged part.
				runs[r].pos = end[r];
				off_t to = (off_t)(end[r] * sizeof(uint64_t)) & ~(off_t)4095;
				if (to > runs[r].punched)
				{
					fallocate(runs[r].fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, runs[r].punched, to - runs[r].punched);
					runs[r].punched = to;
				}
			}
			fprintf(stderr, "absort: merge piece %d: %zu keys from %zu temporary files, %zu unique, %.1f s\n", ++piece, m,
				runs.size(), bk.unique, now() - tp);
		}
		if (!rok)
		{
			fprintf(stderr, "absort: reading a temporary file failed\n");
			return 1;
		}
		for (const Run& r : runs)
		{
			close(r.fd);
		}
		fprintf(stderr, "absort: read %llu pairs (%.1f s), sorted %llu keys of %zu batches (%.1f s), merged them in %.1f s; "
			"%llu duplicates removed, %zu pairs remain", total, t_read, merged_in, runs.size(), t_sort, now() - t3,
			total - (unsigned long long)(unique_keys + wide.size()), unique_keys + wide.size());
		if (!wide.empty())
		{
			fprintf(stderr, " (%zu with |a| of 2^31 or more)", wide.size());
		}
		fprintf(stderr, "\n");
	}
	at = keys_at + (off_t)(unique_keys * AB_RECORD);
	write_wide(wide_low, wide.size());
	if (!to_stdout && close(fd) != 0)
	{
		ok = false;
	}
	if (!ok)
	{
		fprintf(stderr, "absort: writing %s failed\n", outname.c_str());
		return 1;
	}
	free(keys);
	free(sorted);
	const size_t out_pairs = unique_keys + wide.size();
	fprintf(stderr, "absort: wrote %zu pairs (side %d) to %s in %.1f s; total %.1f s\n", out_pairs, side, outname.c_str(),
		now() - t3, now() - t0);
	return 0;
}
