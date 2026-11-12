import sys
def load(fn):
    S = {}
    for line in open(fn):
        if line[0] == '#' or ':' not in line: continue
        s, s0, s1 = line.rstrip().split(':'); a, b = map(int, s.split(','))
        S[(a, b)] = [int(x, 16) for x in s0.split(',')]
    return S
L = load(sys.argv[1]); G = load(sys.argv[2])
qs = sorted({int(l.split('q=')[1].split(';')[0]) for l in open(sys.argv[1]) if l.startswith('# Sieving side-0 q=')})
print(f"las: {len(L)} relations, GPU + NFS_factory: {len(G)}")
print("        q     las    gpu   both  las only  gpu only")
for q in qs:
    l = {k for k, v in L.items() if q in v}; g = {k for k, v in G.items() if q in v}
    print(f"{q:9d}  {len(l):6d} {len(g):6d} {len(l & g):6d}  {len(l - g):6d}  {len(g - l):6d}")
