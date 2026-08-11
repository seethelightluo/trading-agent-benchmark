import re, glob
names = {}
for p in glob.glob('scripts/*.py'):
    s = open(p).read()
    for m in re.findall(r'factor_id[=:]\s*["\']([a-z0-9_]+)["\']', s):
        names.setdefault(m, []).append(p.split('/')[-1])
for k in sorted(names):
    print(k, '->', names[k][0])
