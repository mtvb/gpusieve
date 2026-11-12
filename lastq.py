#!/usr/bin/env python3
"""The special-q of the last pair in a gpusieve .ab file, to resume an interrupted run.

    lastq.py trick.poly file.ab Q0 Q1 [-side S]       (Q0, Q1: the -q0/-q1 of the run; -side as given to gpusieve)

gpusieve sieves the special-q in ascending order and writes the survivors of each one before the next, so the special-q of
the last complete pair is where the run was. Its output may be cut short there: resume with -q0 at that special-q (the
pairs written twice are removed by sort -u). The special-q of a pair is a prime in [Q0, Q1) dividing its value on the
sieved side (Y1*a + Y0*b on side 0, sum c_k a^k b^(d-k) on side 1; as gpusieve, the algebraic side by default if its degree
is 2 or more); as the value can have other prime factors in the range, the one dividing most of the last 50 pairs (all of
its block) is taken. On side 1 a prime can have several special-q (one per root): -q0 at that prime sieves all of them again.
"""
import os, re, struct, sys


def main():
    args = sys.argv[1:]
    side = None
    if '-side' in args:
        k = args.index('-side')
        side = int(args[k + 1])
        del args[k:k + 2]
    if len(args) != 4:
        sys.exit(__doc__)
    poly, ab, q0, q1 = args[0], args[1], int(args[2]), int(args[3])
    Y, c = {}, {}
    for line in open(poly):
        m = re.match(r'\s*([Yc])(\d)\s*:\s*(-?\d+)', line)
        if m:
            (Y if m.group(1) == 'Y' else c)[int(m.group(2))] = int(m.group(3))
    deg = max((k for k, v in c.items() if v), default=-1)
    if side is None:
        side = 1 if deg >= 2 else 0
    if side == 0:
        if set(Y) != {0, 1}:
            sys.exit(f'{poly}: no Y0/Y1 lines')
        value = lambda a, b: Y[1] * a + Y[0] * b
    else:
        if deg < 1:
            sys.exit(f'{poly}: no algebraic side')
        value = lambda a, b: sum(c.get(k, 0) * a ** k * b ** (deg - k) for k in range(deg + 1))
    size = os.path.getsize(ab)
    n = (size - 16) // 12
    if n < 1:
        sys.exit(f'{ab}: no pairs')
    with open(ab, 'rb') as f:
        if f.read(4) != b'NFAB':
            sys.exit(f'{ab}: not an .ab file')
        last = min(n, 50)
        f.seek(16 + 12 * (n - last))
        pairs = list(struct.iter_unpack('<qI', f.read(12 * last)))
    vals = [abs(value(a, b)) for a, b in pairs]
    a, b = pairs[-1]
    v = vals[-1]
    # primes in [q0, q1) by a segmented sieve, tested against v
    small = bytearray([1]) * (int(q1 ** 0.5) + 2)
    small[0:2] = b'\0\0'
    for p in range(2, int(len(small) ** 0.5) + 1):
        if small[p]:
            small[p * p::p] = bytearray(len(small[p * p::p]))
    seg = bytearray([1]) * (q1 - q0)
    for p in (p for p in range(2, len(small)) if small[p]):
        start = max(p * p, (q0 + p - 1) // p * p)
        seg[start - q0::p] = bytearray(len(seg[start - q0::p]))
    found = [q0 + k for k in range(len(seg)) if seg[k] and q0 + k > 1 and v % (q0 + k) == 0]
    if not found:
        sys.exit(f'{ab}: no prime in [{q0}, {q1}) divides the side-{side} value of the last pair ({a},{b}): wrong range, poly file or side?')
    share = {q: sum(x % q == 0 for x in vals) for q in found}
    q = max(found, key=lambda c: share[c])
    print(f'{ab}: {n} pairs; the last one ({a},{b}) belongs to special-q {q} (side {side}; it divides {share[q]} of the last {len(vals)} pairs)')
    others = [c for c in found if c != q]
    if others:
        print(f'   other primes of the range dividing that value: {others} ({[share[c] for c in others]} of the last pairs)')
    if size != 16 + 12 * n:
        print(f'the file ends with {size - 16 - 12 * n} bytes of an incomplete pair')


if __name__ == '__main__':
    main()
