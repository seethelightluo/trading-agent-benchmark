import re
for f in ['scripts/miner_2_20340501_explore_batch1.py', 'scripts/miner_2_20340515_explore_batch2.py']:
    txt = open(f).read()
    print('='*20, f)
    # print factor names mentioned as eval_factor calls
    calls = re.findall(r'eval_factor\(\s*"([^"]+)"', txt)
    print('eval_factor candidates:', calls)
    print(txt[:1500])
    print()
