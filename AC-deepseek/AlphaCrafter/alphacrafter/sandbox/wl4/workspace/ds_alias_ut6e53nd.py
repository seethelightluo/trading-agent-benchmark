import re
for f in ['scripts/miner_1_20281009_explore_batchR.py','scripts/miner_1_20280814_explore_batchQ.py']:
    txt = open(f).read()
    print('='*20, f)
    print('\n'.join(txt.split('\n')[:45]))