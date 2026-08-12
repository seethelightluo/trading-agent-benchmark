import re
for f in ['scripts/miner_2_20280327_explore_batchS.py','scripts/miner_2_20280703_screen_batchQ.py','scripts/miner_2_20280814_screen_batchT.py']:
    try:
        txt = open(f).read()
    except Exception as e:
        print('ERR', f, e); continue
    print('='*25, f)
    for line in txt.split('\n'):
        s = line.strip()
        if re.match(r"^[A-Za-z_]+\[['\"]", s) and ('=' in s) and ('beta' in s or 'mom' in s or 'vol' in s or 'corr' in s or 'ratio' in s or 'z' in s or 'skew' in s or 'rsi' in s or 'csi' in s or 'trend' in s):
            print(s[:110])
    print()