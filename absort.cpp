// absort: sorts the pairs of .ab files and of text files of "a,b" lines numerically by (a, b) (a first, then b, both
// ascending), removes duplicates and writes one .ab file (NFS_factory's format of only a and b, as written by gpusieve,
// RelationsToBinaryFormat and absort itself). Sorted this way, an .ab file compresses best with xz:
//
//     xz -T0 -k --block-size=128MiB --delta=dist=12 --lzma2=preset=6,lc=1,lp=2,pb=2 store.ab
//
// (xz -d or pixz -d return the .ab unchanged.)
//
//     absort [-side 0|1] [-threads N] -o out.ab in1 [in2 ...]
//
// Inputs: .ab files (recognized by their header) and text files whose lines start with "a,b": NFS_factory's
// "format: keys", ab2keys output, or relations "a,b:..." (the rest of a line after b is ignored). Lines that do not start
// with a pair are counted and skipped. Inputs must be regular files (they are mapped into memory). -o - writes to
// standard output.
// -side: the side in the header of the output (0 rational, 1 algebraic). Default: the side of the .ab inputs (they must
// agree), or 0 if all inputs are text.
// Memory: 16 bytes per pair read (two arrays of 64-bit keys), plus the page cache of the inputs.
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
#include <sys/mman.h>
#include <sys/stat.h>

namespace {

const size_t AB_HEADER = 16;
const size_t AB_RECORD = 12;
const int BUCKET_BITS = 16;
const size_t NB = (size_t)1 << BUCKET_BITS;

// Pairs with -2^31 <= a < 2^31 are kept as one 64-bit key whose unsigned order is the order of (a, b).
inline uint64_t make_key(int64_t a, uint32_t b) { return ((uint64_t)(uint32_t)(a + 0x80000000LL) << 32) | b; }
inline int64_t key_a(uint64_t k) { return (int64_t)(k >> 32) - 0x80000000LL; }
inline uint32_t key_b(uint64_t k) { return (uint32_t)k; }
inline bool fits_key(int64_t a) { return a >= -0x80000000LL && a < 0x80000000LL; }

// Pairs with a outside that range (none in practice) are kept apart.
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
};

struct Chunk
{
	int input;
	size_t begin, end;      // bytes of the file (text) or records (.ab)
	size_t bound;           // most pairs it can hold: its lines or records
	size_t offset;          // where its keys go
	size_t count;           // keys it actually held
};

double now()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

int usage()
{
	fprintf(stderr, "usage: absort [-side 0|1] [-threads N] -o out.ab in1 [in2 ...]\n"
		"  sorts the pairs of .ab and text (\"a,b\" lines) files by (a, b), removes duplicates, writes one .ab file\n");
	return 1;
}

} // namespace

int main(int argc, char** argv)
{
	int side = -1, threads = omp_get_max_threads();
	std::string outname;
	std::vector<Input> inputs;
	for (int i = 1; i < argc; i++)
	{
		std::string s = argv[i];
		if (s == "-side" && i + 1 < argc)
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
	omp_set_num_threads(threads);
	double t0 = now();

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
			const size_t per = (size_t)8 << 20;
			for (size_t r = 0; r < records; r += per)
			{
				chunks.push_back({ f, r, std::min(records, r + per), 0, 0, 0 });
			}
		}
		else
		{
			const size_t per = (size_t)128 << 20;
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
	}
	if (side < 0)
	{
		side = ab_side >= 0 ? ab_side : 0;
	}

	// Bounds of the chunks: lines or records.
#pragma omp parallel for schedule(dynamic, 1)
	for (size_t c = 0; c < chunks.size(); c++)
	{
		Chunk& ch = chunks[c];
		const Input& in = inputs[ch.input];
		ch.bound = in.ab ? ch.end - ch.begin : count_lines(in.data + ch.begin, ch.end - ch.begin);
	}
	size_t bound = 0;
	for (Chunk& ch : chunks)
	{
		ch.offset = bound;
		bound += ch.bound;
	}
	fprintf(stderr, "absort: up to %zu pairs in %zu input files, %.1f GB of memory for the keys, %d threads\n", bound,
		inputs.size(), 16.0 * (double)bound / 1e9, threads);

	uint64_t* keys = (uint64_t*)malloc(std::max<size_t>(bound, 1) * sizeof(uint64_t));
	if (!keys)
	{
		fprintf(stderr, "absort: cannot allocate %.1f GB\n", 8.0 * (double)bound / 1e9);
		return 1;
	}

	// Read the pairs.
	std::vector<std::vector<Wide>> wide_t(threads);
	std::vector<unsigned long long> pairs_c(chunks.size()), bad_c(chunks.size());
#pragma omp parallel for schedule(dynamic, 1)
	for (size_t c = 0; c < chunks.size(); c++)
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
	for (size_t c = 0; c < chunks.size(); c++)
	{
		Chunk& ch = chunks[c];
		if (n != ch.offset)
		{
			memmove(keys + n, keys + ch.offset, ch.count * sizeof(uint64_t));
		}
		n += ch.count;
		inputs[ch.input].pairs += pairs_c[c];
		inputs[ch.input].bad += bad_c[c];
	}
	std::vector<Wide> wide;
	for (auto& w : wide_t)
	{
		wide.insert(wide.end(), w.begin(), w.end());
		std::vector<Wide>().swap(w);
	}
	unsigned long long total = 0;
	for (const Input& in : inputs)
	{
		fprintf(stderr, "absort: %s: %s, %llu pairs", in.name.c_str(), in.ab ? ".ab" : "text", in.pairs);
		if (in.bad)
		{
			fprintf(stderr, ", %llu lines without a pair skipped", in.bad);
		}
		fprintf(stderr, "\n");
		total += in.pairs;
		if (in.data)
		{
			munmap((void*)in.data, in.size);
		}
	}
	double t1 = now();
	fprintf(stderr, "absort: read %llu pairs in %.1f s\n", total, t1 - t0);

	// Bucket the keys by their top bits into a second array, then sort and deduplicate each bucket.
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
	uint64_t* sorted = (uint64_t*)malloc(std::max<size_t>(n, 1) * sizeof(uint64_t));
	if (!sorted)
	{
		fprintf(stderr, "absort: cannot allocate %.1f GB\n", 8.0 * (double)n / 1e9);
		return 1;
	}
	std::vector<size_t> hist((size_t)threads * NB, 0);
	std::vector<size_t> bstart(NB + 1, 0), bsize(NB, 0);
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
				bstart[b] = pos;
				for (int u = 0; u < nt; u++)
				{
					size_t c = hist[(size_t)u * NB + b];
					hist[(size_t)u * NB + b] = pos;
					pos += c;
				}
			}
			bstart[NB] = pos;
		}
		for (size_t i = lo; i < hi; i++)
		{
			sorted[h[(keys[i] - kmin) >> shift]++] = keys[i];
		}
	}
	free(keys);
	double t2 = now();
#pragma omp parallel for schedule(dynamic, 16)
	for (size_t b = 0; b < NB; b++)
	{
		uint64_t* s = sorted + bstart[b];
		uint64_t* e = sorted + bstart[b + 1];
		std::sort(s, e);
		bsize[b] = (size_t)(std::unique(s, e) - s);
	}
	std::sort(wide.begin(), wide.end());
	wide.erase(std::unique(wide.begin(), wide.end()), wide.end());
	size_t wide_low = (size_t)(std::partition_point(wide.begin(), wide.end(), [](const Wide& w) { return w.a < 0; }) - wide.begin());
	std::vector<size_t> ostart(NB + 1, 0);
	for (size_t b = 0; b < NB; b++)
	{
		ostart[b + 1] = ostart[b] + bsize[b];
	}
	const size_t unique_keys = ostart[NB], out_pairs = unique_keys + wide.size();
	double t3 = now();
	fprintf(stderr, "absort: bucketed in %.1f s, sorted in %.1f s; %llu duplicates removed, %zu pairs remain", t2 - t1, t3 - t2,
		total - (unsigned long long)out_pairs, out_pairs);
	if (!wide.empty())
	{
		fprintf(stderr, " (%zu with |a| of 2^31 or more)", wide.size());
	}
	fprintf(stderr, "\n");

	// Write: header, pairs with a < -2^31, the buckets in order, pairs with a >= 2^31.
	const bool to_stdout = outname == "-";
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
	const size_t piece = (size_t)1 << 20;
	if (positioned)
	{
		// Each thread converts whole buckets and writes them at their place in the file.
		bool wok = true;
#pragma omp parallel reduction(&& : wok)
		{
			std::vector<char> buf(piece * AB_RECORD);
#pragma omp for schedule(dynamic, 16)
			for (size_t b = 0; b < NB; b++)
			{
				const uint64_t* s = sorted + bstart[b];
				for (size_t i = 0; i < bsize[b]; i += piece)
				{
					size_t m = std::min(piece, bsize[b] - i);
					for (size_t j = 0; j < m; j++)
					{
						put_record(&buf[j * AB_RECORD], key_a(s[i + j]), key_b(s[i + j]));
					}
					wok = write_all(fd, buf.data(), m * AB_RECORD, keys_at + (off_t)((ostart[b] + i) * AB_RECORD), true) && wok;
				}
			}
		}
		ok = ok && wok;
	}
	else
	{
		std::vector<char> buf(piece * AB_RECORD);
		size_t m = 0;
		for (size_t b = 0; b < NB && ok; b++)
		{
			const uint64_t* s = sorted + bstart[b];
			for (size_t i = 0; i < bsize[b]; i++)
			{
				put_record(&buf[m * AB_RECORD], key_a(s[i]), key_b(s[i]));
				if (++m == piece)
				{
					ok = ok && write_all(fd, buf.data(), m * AB_RECORD, 0, false);
					m = 0;
				}
			}
		}
		ok = ok && write_all(fd, buf.data(), m * AB_RECORD, 0, false);
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
	free(sorted);
	fprintf(stderr, "absort: wrote %zu pairs (side %d) to %s in %.1f s; total %.1f s\n", out_pairs, side, outname.c_str(),
		now() - t3, now() - t0);
	return 0;
}
