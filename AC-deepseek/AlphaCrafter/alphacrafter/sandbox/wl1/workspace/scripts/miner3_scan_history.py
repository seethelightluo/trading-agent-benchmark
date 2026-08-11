import re, glob
for fn in sorted(glob.glob('scripts/miner3_*.py')):
    try:
        txt = open(fn).read()
        names = re.findall(r'["\']([a-z0-9_]+)["\']\s*[:=]\s*(?:C|factors|lr|vol|beta|panel)', txt)
        if names:
            print(fn.split('/')[-1], '->', list(dict.fromkeys(names))[:20])
    except Exception as e:
        print(fn, 'ERR', e)
