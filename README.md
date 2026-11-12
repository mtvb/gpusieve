# gpusieve

gpusieve is a special-q lattice siever for NVIDIA GPUs. It sieves **one side** of a number field sieve polynomial pair:
the side that many numbers share in a factorization factory. The result is a store of (a,b) pairs that are smooth on the
shared side. [NFS_factory](https://github.com/mtvb/NFS_factory) then checks the store against the other side of each
number, and writes the relations for msieve or CADO-NFS.

The shared side can be

- a **rational side** `Y1*x + Y0` (side 0), shared by numbers that use the same rational polynomial, or
- an **algebraic side** of degree up to 8 (side 1), shared by a family of numbers (SNFS).

gpusieve uses the conventions of CADO-NFS `las` (lattice reduction, sieve region, special-q), so on the same special-q it
finds the same relations as las.

## Requirements

- An NVIDIA GPU of compute capability 7.0 or later. GPUs with less memory sieve each special-q in more
  passes (see [GPU memory](#gpu-memory)).
- Linux, the CUDA Toolkit, GCC with OpenMP, and make.
- NFS_factory from the fork linked above, which has `format: keys` and reads `.ab` files from standard input.

## Build

```
make            gpusieve, absort and ab2keys
make tools      only absort and ab2keys (no CUDA needed)
```

| Setting | Meaning |
|---|---|
| `CUDA_ARCH` | `native` (default): the GPUs of the build machine. `all`: every GPU nvcc supports, for a binary to run elsewhere. A list such as `"75 86 120"`: those, plus PTX for the last. |
| `NVCC_CCBIN` | Host compiler for nvcc, if the default GCC is too new for the CUDA version, e.g. `NVCC_CCBIN=g++-13` for CUDA 12.4. |
| `NVCC` | The CUDA compiler (default `nvcc`). |
| `OPT` | Host optimization flags (default `-O3 -march=native`, so build on the machine that runs the programs). |

With `CUDA_ARCH=native` and no GPU visible during the build, nvcc only warns and builds for an old default architecture;
use `CUDA_ARCH=all` there.

## Building a store

The example builds a store for a shared rational side `x - m`. For an algebraic store, swap the roles of the sides as
described in [Algebraic store](#algebraic-store).

### 1. Poly files

gpusieve reads a CADO-NFS poly file. The side that is not sieved must be linear; a **dummy** side `2x - 1` stands in for
it. `sieve.poly`:

```
n: <|2m - 1|, the resultant of the two sides>
skew: <skewness of the sieve region>
c0: -1
c1: 2
Y0: -<m>
Y1: 1
```

For a rational store, choose `skew` like the skewness of the numbers' algebraic polynomials.

`check.poly` for NFS_factory is the same file with the bounds of the shared side and two more lines:

```
rlim: 25000000
lpbr: 28
mfbr: 56
alim: 1000000
lpba: 32
mfba: 32
side: 0
format: keys
```

`side` is the side being checked, here the rational one. `format: keys` writes only `a,b` for each smooth pair.
`mfbr = 2*lpbr` keeps pairs with up to two large primes; `3*lpbr` keeps those with three as well.

### 2. Sieve

```
gpusieve -poly sieve.poly -side 0 -I 16 -lim 268435456 -mfb 0 -slack 16 -q0 Q0 -q1 Q1 -out part1.ab
```

`-lim` is `2^lpbr` here (`2^28`): see [Parameters](#parameters). gpusieve prints a progress line every minute. If a run is
interrupted, `lastq.py sieve.poly part1.ab Q0 Q1` gives the special-q to continue from with `-q0`.

### 3. Check

```
NFS_factory_gpu check.poly part1.ab part1.txt
```

### 4. Sort into a store

```
absort -side 0 -o store.ab part1.txt part2.txt ...
```

absort sorts all pairs, removes duplicates (a pair is found by every special-q that divides its value), and writes one
`.ab` file. `-side 0` marks the store as smooth on side 0.

### 5. Use the store for a number

`number.poly` holds the number's own polynomial pair: its algebraic side, and the shared rational side. Add
`side: 1` (the side to check now), the bounds of both sides, and `format: msieve` or `format: cado`:

```
NFS_factory_gpu number.poly store.ab rels.txt
```

### 6. Share the store

Sorted by absort, an `.ab` file compresses best with xz's delta filter (about 28 bits per pair), and stays a plain `.ab`
file when unpacked:

```
xz -T0 -k --block-size=128MiB --delta=dist=12 --lzma2=preset=6,lc=1,lp=2,pb=2 store.ab
xz -T0 -dc store.ab.xz | NFS_factory_gpu number.poly - rels.txt
```

### Algebraic store

For a shared algebraic side, `sieve.poly` holds the shared polynomial (`c0` ... `cd`, its `skew`) and the dummy
rational side `Y1: 2`, `Y0: -1`, with `n` the resultant `|2^d * f(1/2)|`. Sieve with `-side 1` and
`-lim 2^lpba`. `check.poly` has `side: 1` and the bounds of the algebraic side. Build the store with `absort -side 1`.
Each number's poly file then holds the shared polynomial and the number's own `Y0`, `Y1`, with `side: 0`.

### Which side the header names

An `.ab` file's header names one side, and NFS_factory refuses to check that side:

- gpusieve writes the side that is not sieved, so that the check of the sieved side can run.
- `absort -side S` writes S, the side the store's pairs are smooth on, so that each number's check runs on the other one.

If NFS_factory says the relations "were sieved for side S, which is the side to check here", set byte 8 of the file to
the other side (here to 0):

```
printf '\000' | dd of=file.ab bs=1 seek=8 count=1 conv=notrunc
```

## Parameters

**Sieve to the large-prime bound.** Use `-lim 2^lpb -mfb 0` with a slack of 16 to 20 bits (16 for lpb 28, 20 for
lpb 29). With every prime up to the large-prime bound in the factor base, the part of a value that the sieve does not
account for is either small (prime powers counted once, rounding) or at least lpb bits (a prime above the bound). A
slack between the two keeps almost only smooth pairs: 60-95 % of the survivors pass the check, depending on how many large primes it
allows. The classic
mode, with `-lim` below `2^lpb` and `-mfb 2*lpb`, writes 15-20 times as many survivors, most of them not smooth.

**Sieve region.** Each step up in `-I` takes about 4 times as long per special-q and gives 1.8 (degree 7) to 3.6
(rational side) times as many relations, so the relations per GPU hour drop with it, the more so for high degrees.
`-I 16` and `-I 17` are the usual choices; up to `-I 19` works. A larger `-I` needs no larger GPU: gpusieve sieves a
special-q in as many passes as needed (see [GPU memory](#gpu-memory)).

**Special-q range.** Small special-q give the smallest pairs, which are the most useful ones, but more duplicates and
many survivors per special-q (raise `-maxout` if the run reports dropped survivors). `-random N` sieves N random
special-q of the range, to estimate the yield before a long run.

**Special-q above the check's lim.** If special-q go above the `rlim` (or `alim`) of the check, the special-q itself
counts as a large prime there. Then use a lim of at least the largest special-q in the check, or `mfb = 3*lpb`.

## Options

| Option | Meaning | Default |
|---|---|---|
| `-poly FILE` | CADO-NFS poly file | |
| `-side 0\|1` | side to sieve | algebraic if its degree is 2 or more, else rational |
| `-I N` | sieve region of `2^N` by up to `2^(N-1)`, N from 10 to 19 | 15 |
| `-parts N` | sieve each special-q in N passes over its rows, N a power of 2 | as few as fit into the GPU's memory and L2 cache |
| `-lim L` | factor base bound (below `2^32`) | 25000000 |
| `-mfb B` | bits of the value allowed beyond the factor base | 56 |
| `-slack S` | extra bits for prime powers and rounding | 12 |
| `-q0 Q0 -q1 Q1` | special-q range `[Q0, Q1)`, on the sieved side | |
| `-random N` | only N random special-q of the range | |
| `-qfile FILE` | special-q, basis and J from las output (`# Sieving side-S q=...` lines), for comparisons with las | |
| `-out FILE` | output `.ab` file | |
| `-dummybits B` | a dummy side's values must stay below `2^B`; 0: not checked | 32 |
| `-maxout M` | at most `M * 2^20` survivors per special-q (32 MB of GPU and 32 MB of host memory per unit); more are counted and reported | 16 |
| `-threads N` | host threads for the factor base and the special-q roots | 16 |
| `-progress S` | a progress line every S seconds, 0 for none | 60 |
| `-v` | timings of each special-q | |

Tuning options, rarely needed: `-tiny P` (primes below P are sieved by all threads of a region, default 64),
`-lanes 2` (fill the buckets of the next special-q while the rows of this one are sieved), `-bblocks N` and
`-rblocks N` (persistent grids for the two kernels), `-twopass` (another bucket kernel), and `-dumpfb FILE` (write the
factor base and stop).

## GPU memory

The buckets hold the hits of the large primes. For a whole special-q they grow with the sieve region and the factor
base bound:

| `-I` | `-lim` | buckets of a whole special-q |
|---|---|---|
| 15 | 25000000 | 1.6 GB |
| 16 | 25000000 | 5.3 GB |
| 15 | `2^29` | 2.3 GB |
| 17 | `2^28` | 24.3 GB |
| 17 | `2^29` | 25.5 GB |
| 18 | `2^28` | 85 GB |
| 19 | `2^28` | 296 GB |

gpusieve sieves each special-q in N passes over its rows (N a power of 2) and holds the buckets of one pass at a time,
1/N of the value above. It chooses N so that they fit into the free GPU memory, and so that a pass has at most about
L2/256 bucket regions: the ends of the buckets being filled then stay in the GPU's L2 cache, which makes the bucket
fill faster (with a 4.5 MB L2, 16 passes at `-I 17` were 1.9 times as fast as 2). `-parts N` sets N.

Besides one pass' buckets, the GPU holds the factor base (about 30 bytes per entry, 0.4 GB at lim `2^28` on a rational
side, 0.8 GB at lim `2^29` on a side of degree 7) and two output buffers (32 MB per `-maxout` unit, 512 MB by default).
At startup gpusieve prints the number of passes and the memory it needs, and stops if the GPU does not have it.

## Output format (.ab)

A 16-byte header (`"NFAB"`, then version 1, side and flags as 32-bit integers), then 12 bytes per pair: `a` as a 64-bit
and `b` as a 32-bit integer, little-endian. This is NFS_factory's format of only a and b. gpusieve writes the survivors
of each special-q before the next, in ascending order of special-q, and does not remove duplicates between special-q.

## Tools

| Tool | Purpose |
|---|---|
| `absort [-side S] [-threads N] -o out.ab in ...` | Sorts `.ab` files and text files of `a,b` lines by (a, b), removes duplicates, and writes one `.ab` file (`-o -` for standard output). Needs 16 bytes of memory per pair. |
| `ab2keys file.ab ...` | Writes the pairs of `.ab` files as `a,b` lines. |
| `lastq.py poly file.ab Q0 Q1 [-side S]` | The special-q of the last pair of an interrupted run, to continue with `-q0`. |
| `lasnorm.pl` | Rewrites CADO-format relations as las prints them, so that `sort -u` removes relations found by both. |
| `factorypoly.py N m [options]` | Searches for algebraic polynomials of a number N with a fixed rational side `x - m`, rated with CADO-NFS (`-cado` names its polyselect folder). |

## Validation

On the same special-q as CADO-NFS las, gpusieve with the NFS_factory check finds las' relations (rational sides at
I = 15 and 16, algebraic sides of degree 6 and 7 at I = 15), except a few in 100,000. Those contain a prime power
that the sieve counts only once, and a larger slack keeps most of them. It also finds a few relations that las misses.
The lattice bases and J are the same as las'. `test/` holds poly files and las outputs of these comparisons, and
`compare.py`, which compares two relation sets per special-q.

## Limitations

- The poly file's other side, which gpusieve does not sieve, must be linear: a dummy such as `2x - 1` (see
  [Poly files](#1-poly-files)). A polynomial of degree 2 or more there is refused.
- Degree up to 8, `-I` up to 19, `-lim` below `2^32`.
- `b` must stay below `2^32` (it has 32 bits in `.ab` files, NFS_factory and msieve). Larger `b` occur at large `-I`
  with large special-q (`-I 19`: above about `10^8`); such survivors are left out and counted at the end of the run.
- Prime powers above `2^I` are not sieved; the slack covers most of them.
- Tested on Linux only (absort maps its inputs into memory).

## Authors and license

Markus Tervooren with the help of AI (Claude). MIT license, see `LICENSE`.
