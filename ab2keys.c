// ab2keys: writes the pairs of .ab files (NFS_factory's format of only a and b, as written by gpusieve and by
// RelationsToBinaryFormat) as text lines "a,b", the same as NFS_factory's "format: keys" and as the first field of a
// CADO-NFS relation (cut -d: -f1), for sort -u and merging.
//
//     ab2keys file.ab [file2.ab ...] > keys.txt          (- reads standard input)
//
// Build: gcc -O2 -o ab2keys ab2keys.c
// .ab: a header of 16 bytes ("NFAB", (uint32) version 1, (uint32) side, (uint32) flags), then (int64) a, (uint32) b per
// pair, little-endian.
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static char out[1 << 20];
static size_t used;

static void flush(void) { fwrite(out, 1, used, stdout); used = 0; }

static char* put_u64(char* p, uint64_t v)
{
	char tmp[20]; int n = 0;
	do { tmp[n++] = (char)('0' + v % 10); v /= 10; } while (v);
	while (n) *p++ = tmp[--n];
	return p;
}

static int convert(const char* name)
{
	FILE* f = strcmp(name, "-") ? fopen(name, "rb") : stdin;
	if (!f) { fprintf(stderr, "ab2keys: cannot open %s\n", name); return 1; }
	unsigned char h[16];
	if (fread(h, 1, 16, f) != 16 || memcmp(h, "NFAB", 4) != 0 || h[4] != 1) {
		fprintf(stderr, "ab2keys: %s is not an .ab file of version 1\n", name);
		if (f != stdin) fclose(f);
		return 1;
	}
	static unsigned char in[12 << 16];
	size_t got, rest = 0;
	unsigned long long n = 0;
	while ((got = fread(in + rest, 1, sizeof in - rest, f)) > 0) {
		size_t len = rest + got, k = 0;
		for (; k + 12 <= len; k += 12) {
			int64_t a; uint32_t b;
			memcpy(&a, in + k, 8); memcpy(&b, in + k + 8, 4);
			if (used > sizeof out - 64) flush();
			char* p = out + used;
			if (a < 0) { *p++ = '-'; p = put_u64(p, (uint64_t)0 - (uint64_t)a); } else p = put_u64(p, (uint64_t)a);
			*p++ = ',';
			p = put_u64(p, b);
			*p++ = '\n';
			used = (size_t)(p - out);
			n++;
		}
		rest = len - k;
		memmove(in, in + k, rest);
	}
	if (f != stdin) fclose(f);
	if (rest) fprintf(stderr, "ab2keys: %s ends with %zu bytes of an incomplete pair\n", name, rest);
	fprintf(stderr, "ab2keys: %s: %llu pairs\n", name, n);
	return rest != 0;
}

int main(int argc, char** argv)
{
	if (argc < 2) { fprintf(stderr, "usage: ab2keys file.ab [file2.ab ...] > keys.txt   (- reads standard input)\n"); return 1; }
	int status = 0;
	for (int k = 1; k < argc; k++) status |= convert(argv[k]);
	flush();
	return status;
}
