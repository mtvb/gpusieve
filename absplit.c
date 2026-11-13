// absplit: splits the pairs of .ab files by b. Pairs with b < B go to one .ab file, the others to another, in the order
// they are read. The inputs are streamed, so a store of any size can be split, also from a pipe (xz -dc store.ab.xz |).
//
//     absplit [-b B] [-side S] [-small small.ab] [-large large.ab] in1.ab [in2.ab ...]       (- reads standard input)
//
// -b: the bound (default 1048576 = 2^20). Pairs with b < B go to -small, the others to -large. At least one of the two
//     outputs must be given; the pairs of a part without an output are only counted.
// -side: the side in the headers of the outputs (0 or 1). Default: the side of the inputs, which must then agree.
//
// Use: survivors of gpusieve versions up to f764f7a contain the lattice basis vector (a0,b0) of each special-q without a
// smoothness test. On a sieved side such as x - m with a large m its value is about m*b0, almost never smooth. It is a
// short vector, so b0 is small: below about the square root of q (at most a few thousand in tests up to q = 7e7). Split
// the survivors at b = 2^20, run only the small part through the store check of NFS_factory, and sort the checked small
// part and the large part together:
//
//     absplit -side 1 -small small.ab -large large.ab surv1.ab surv2.ab ...
//     NFS_factory_gpu storecheck.poly small.ab small.k
//     absort -side 0 -o store.ab small.k large.ab
//
// Build: gcc -O2 -o absplit absplit.c
// .ab: a header of 16 bytes ("NFAB", (uint32) version 1, (uint32) side, (uint32) flags), then (int64) a, (uint32) b per
// pair, little-endian.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#define RECORD 12
#define IN_SIZE ((size_t)RECORD << 20)    // 12 MB of whole records
#define OUT_SIZE ((size_t)RECORD << 19)

typedef struct
{
	const char* name;
	FILE* f;
	unsigned char* buf;
	size_t used;
	unsigned long long pairs;
	int failed;
} Output;

static void out_flush(Output* o)
{
	if (o->f && o->used && !o->failed && fwrite(o->buf, 1, o->used, o->f) != o->used)
	{
		o->failed = 1;
	}
	o->used = 0;
}

static int out_open(Output* o, uint32_t side)
{
	if (!o->name)
	{
		return 0;
	}
	o->f = fopen(o->name, "wb");
	o->buf = malloc(OUT_SIZE);
	if (!o->f || !o->buf)
	{
		fprintf(stderr, "absplit: cannot create %s\n", o->name);
		return 1;
	}
	unsigned char h[16];
	const uint32_t v[3] = { 1, side, 0 };
	memcpy(h, "NFAB", 4);
	memcpy(h + 4, v, 12);
	if (fwrite(h, 1, 16, o->f) != 16)
	{
		o->failed = 1;
	}
	return 0;
}

static int same_file(const char* x, const char* y)
{
	struct stat a, b;
	return stat(x, &a) == 0 && stat(y, &b) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

static int usage(void)
{
	fprintf(stderr, "usage: absplit [-b B] [-side 0|1] [-small small.ab] [-large large.ab] in1.ab [in2.ab ...]   (- reads standard input)\n"
		"  pairs with b < B (default 1048576) go to -small, the others to -large\n");
	return 1;
}

int main(int argc, char** argv)
{
	unsigned long long bound = 1ull << 20;
	int side = -1;
	Output small = { 0 }, large = { 0 };
	const char** inputs = malloc(sizeof(char*) * (size_t)argc);
	int ninputs = 0;
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "-b") && i + 1 < argc)
		{
			char* end;
			bound = strtoull(argv[++i], &end, 10);
			if (*end || bound == 0)
			{
				return usage();
			}
		}
		else if (!strcmp(argv[i], "-side") && i + 1 < argc)
		{
			side = atoi(argv[++i]);
			if (side != 0 && side != 1)
			{
				return usage();
			}
		}
		else if (!strcmp(argv[i], "-small") && i + 1 < argc)
		{
			small.name = argv[++i];
		}
		else if (!strcmp(argv[i], "-large") && i + 1 < argc)
		{
			large.name = argv[++i];
		}
		else if (argv[i][0] == '-' && argv[i][1])
		{
			return usage();
		}
		else
		{
			inputs[ninputs++] = argv[i];
		}
	}
	if (ninputs == 0 || (!small.name && !large.name))
	{
		return usage();
	}
	if (small.name && large.name && (!strcmp(small.name, large.name) || same_file(small.name, large.name)))
	{
		fprintf(stderr, "absplit: -small and -large name the same file\n");
		return 1;
	}
	for (int k = 0; k < ninputs; k++)
	{
		if ((small.name && same_file(inputs[k], small.name)) || (large.name && same_file(inputs[k], large.name)))
		{
			fprintf(stderr, "absplit: %s is an input and an output\n", inputs[k]);
			return 1;
		}
	}

	unsigned char* in = malloc(IN_SIZE + RECORD);
	if (!in)
	{
		fprintf(stderr, "absplit: out of memory\n");
		return 1;
	}
	const time_t t0 = time(NULL);
	int opened = 0, status = 0;
	uint32_t out_side = 0;
	for (int k = 0; k < ninputs && !status; k++)
	{
		const char* name = inputs[k];
		FILE* f = strcmp(name, "-") ? fopen(name, "rb") : stdin;
		if (!f)
		{
			fprintf(stderr, "absplit: cannot open %s\n", name);
			status = 1;
			break;
		}
		unsigned char h[16];
		uint32_t version = 0, in_side;
		const int header_ok = fread(h, 1, 16, f) == 16 && memcmp(h, "NFAB", 4) == 0;
		if (header_ok)
		{
			memcpy(&version, h + 4, 4);
		}
		if (!header_ok || version != 1)
		{
			fprintf(stderr, "absplit: %s is not an .ab file of version 1\n", name);
			status = 1;
			if (f != stdin) fclose(f);
			break;
		}
		memcpy(&in_side, h + 8, 4);
		if (!opened)
		{
			out_side = side >= 0 ? (uint32_t)side : in_side;
			if (out_open(&small, out_side) || out_open(&large, out_side))
			{
				status = 1;
				if (f != stdin) fclose(f);
				break;
			}
			opened = 1;
		}
		else if (side < 0 && in_side != out_side)
		{
			fprintf(stderr, "absplit: %s is of side %u, the inputs before it of side %u; choose one with -side\n", name, in_side,
				out_side);
			status = 1;
			if (f != stdin) fclose(f);
			break;
		}
		unsigned long long n_small = 0, n_large = 0;
		size_t rest = 0, got;
		while ((got = fread(in + rest, 1, IN_SIZE - rest, f)) > 0)
		{
			const size_t len = rest + got, whole = len - len % RECORD;
			for (size_t p = 0; p < whole; p += RECORD)
			{
				uint32_t b;
				memcpy(&b, in + p + 8, 4);
				Output* o = (unsigned long long)b < bound ? &small : &large;
				if (o == &small) n_small++; else n_large++;
				if (o->f)
				{
					if (o->used + RECORD > OUT_SIZE) out_flush(o);
					memcpy(o->buf + o->used, in + p, RECORD);
					o->used += RECORD;
				}
			}
			rest = len - whole;
			memmove(in, in + whole, rest);
		}
		if (ferror(f))
		{
			fprintf(stderr, "absplit: reading %s failed\n", name);
			status = 1;
		}
		if (f != stdin) fclose(f);
		if (rest)
		{
			fprintf(stderr, "absplit: warning: %s ends with %zu bytes of an incomplete pair (ignored)\n", name, rest);
		}
		small.pairs += n_small;
		large.pairs += n_large;
		fprintf(stderr, "absplit: %s: %llu pairs, %llu with b < %llu, %llu with larger b\n", name, n_small + n_large, n_small, bound,
			n_large);
	}
	Output* outs[2] = { &small, &large };
	for (int k = 0; k < 2; k++)
	{
		Output* o = outs[k];
		if (!o->f) continue;
		out_flush(o);
		if (fclose(o->f) != 0) o->failed = 1;
		if (o->failed)
		{
			fprintf(stderr, "absplit: writing %s failed\n", o->name);
			status = 1;
		}
	}
	if (!status)
	{
		fprintf(stderr, "absplit: %llu pairs with b < %llu%s%s, %llu with larger b%s%s (side %u), %ld s\n", small.pairs, bound,
			small.name ? " to " : " (not written)", small.name ? small.name : "", large.pairs, large.name ? " to " : " (not written)",
			large.name ? large.name : "", out_side, (long)(time(NULL) - t0));
	}
	return status;
}
