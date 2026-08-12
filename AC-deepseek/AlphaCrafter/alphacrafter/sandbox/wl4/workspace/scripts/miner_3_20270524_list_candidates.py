import re, glob
cands = {}
for f in sorted(glob.glob('scripts/miner_3_*screen*.py')) + sorted(glob.glob('scripts/miner_3_*explore*.py')):
    txt = open(f).read()
    names = re.findall(r'cands\[\s*["\']([A-Za-z0-9_]+)["\']', txt)
    cands[f.split('/')[-1]] = names
for f, names in cands.items():
    print(f, len(names))
    print('   ', names)
