import re
for fn in ['scripts/miner3_20260716_screen_cycle2.py',
           'scripts/miner3_20260716_screen_novel2.py',
           'scripts/miner3_20260716_screen_novel3.py',
           'scripts/miner3_20260716_screen_cycle7.py',
           'scripts/miner3_20260716_screen_cycle8_macro_vol.py']:
    txt = open(fn).read()
    print('=' * 20, fn.split('/')[-1])
    # print lines that assign factor dict entries or define factor names
    for line in txt.splitlines():
        if re.search(r'factors\[|"f_|cand|CAND|name\s*=|dict\(', line):
            print(line.strip()[:150])
