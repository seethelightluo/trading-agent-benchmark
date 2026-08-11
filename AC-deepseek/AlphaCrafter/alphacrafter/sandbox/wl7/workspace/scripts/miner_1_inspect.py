import glob, re

for f in sorted(glob.glob('scripts/miner_3_*.py')):
    txt = open(f).read()
    ids = set(re.findall(r'factor_id\s*=\s*["\']([^"\']+)["\']', txt))
    ids |= set(re.findall(r'["\'](?:factor_ids?)["\']\s*:\s*\[([^\]]+)\]', txt))
    names = set(re.findall(r'(?:def factor_|name\s*=\s*|"name"\s*:\s*)"?([a-z][a-z0-9_]+)"?', txt))
    print('==', f.split('/')[-1])
    print('  ids:', sorted(ids)[:20])
