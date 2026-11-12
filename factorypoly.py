#!/usr/bin/env python3
"""GNFS polynomials for a number N with a FIXED rational side x - m (the shared side of a factorization factory).

    factorypoly.py N m [-d 5] [-K 4] [-H 1] [-Bf 1.4e7] [-Bg 1.1e7] [-area 1e16] [-keep 3] [-j 16] [-cado DIR] [-out f.poly]

Every polynomial f of degree d with f(m) = k*N has the root m modulo N, so (f, x - m) is a GNFS pair for N. Candidates:
  - multipliers k = 1..K (f(m) = k*N), each written in base m with digits in (-m/2, m/2] (the leading digit must stay
    below m: K is limited to about m^(d+1) / (2N));
  - f + h(x) * (x - m) for every h of degree < d with coefficients in [-H, H] (rotation by linear h is the usual
    rotation; the higher ones move the other coefficients). Each step changes coefficients by about m, so only small H
    makes sense.
Each candidate is rated with CADO's skewness and score (Murphy E for Bf, Bg, area; the defaults are CADO's c140 lims and
the default area); the best ones are printed as CADO poly files with the rational side Y1 = 1, Y0 = -m unchanged.
"""
import sys, os, math, itertools, subprocess, tempfile, concurrent.futures

def main():
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    N, m = int(a[0]), int(float(a[1])) if 'e' in a[1].lower() else int(a[1])
    opt = {'-d': '5', '-K': '4', '-H': '1', '-Bf': '1.4e7', '-Bg': '1.1e7', '-area': '1e16', '-keep': '3', '-j': '16',
           '-cado': '/root/cado-nfs/build/s140/polyselect', '-out': ''}
    for i in range(2, len(a), 2):
        if a[i] not in opt:
            sys.exit(f'unknown option {a[i]}')
        opt[a[i]] = a[i + 1]
    d, K, H, keep, j = int(opt['-d']), int(opt['-K']), int(opt['-H']), int(opt['-keep']), int(opt['-j'])
    cado = opt['-cado']
    sc = ['-Bf', opt['-Bf'], '-Bg', opt['-Bg'], '-area', opt['-area']]

    def basem(x):
        c = []
        for _ in range(d):
            t = x % m
            if t > m // 2:
                t -= m
            c.append(t); x = (x - t) // m
        c.append(x)
        return c

    cands = []
    for k in range(1, K + 1):
        f0 = basem(k * N)
        if abs(f0[d]) > m:
            break                                   # k*N needs a higher degree with this m
        for h in itertools.product(range(-H, H + 1), repeat=d):
            f = list(f0)
            for i, hi in enumerate(h):              # + hi * x^i * (x - m)
                f[i + 1] += hi; f[i] -= hi * m
            if f[d] == 0:
                continue
            g = 0
            for c in f:
                g = math.gcd(g, c)
            if g > 1:
                f = [c // g for c in f]
            cands.append((k, h, f))
    print(f'{len(cands)} candidates (k <= {K}, h coefficients in [{-H},{H}])', file=sys.stderr)

    tmp = tempfile.mkdtemp(prefix='factorypoly')

    def polytext(f, skew=None):
        s = f'n: {N}\n' + (f'skew: {skew}\n' if skew else '')
        s += ''.join(f'c{i}: {f[i]}\n' for i in range(d, -1, -1))
        return s + f'Y1: 1\nY0: {-m}\n'

    def rate(t):
        k, h, f = cands[t]
        p = os.path.join(tmp, f'{t}.poly')
        open(p, 'w').write(polytext(f))
        r = subprocess.run([cado + '/skewness', p], capture_output=True, text=True).stdout.split()
        skew = float(r[-1])
        open(p, 'w').write(polytext(f, f'{skew:.6g}'))
        e = subprocess.run([cado + '/score'] + sc + [p], capture_output=True, text=True).stdout.split()
        os.remove(p)
        try:
            return float(e[-1]), skew, t
        except ValueError:
            return 0.0, skew, t

    with concurrent.futures.ThreadPoolExecutor(j) as ex:
        res = sorted(ex.map(rate, range(len(cands))), reverse=True)
    os.rmdir(tmp)
    base = [r for r in res if cands[r[2]][0] == 1 and not any(cands[r[2]][1])]
    if base:
        print(f'plain base-m polynomial (k = 1, h = 0): E = {base[0][0]:.4g}', file=sys.stderr)
    out = []
    for E, skew, t in res[:keep]:
        k, h, f = cands[t]
        out.append(f'# MurphyE = {E:.4g} (Bf {opt["-Bf"]}, Bg {opt["-Bg"]}, area {opt["-area"]}), k = {k}, h = {h}\n' + polytext(f, f'{skew:.6g}'))
    text = '\n'.join(out)
    if opt['-out']:
        open(opt['-out'], 'w').write(text)
    print(text)

if __name__ == '__main__':
    main()
