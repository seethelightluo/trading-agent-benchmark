import re
for f in ['scripts/miner_1_20280814_explore_batchQ.py','scripts/miner_1_20281009_explore_batchR.py','scripts/miner_1_20290115_explore_batchS.py']:
    txt = open(f).read()
    print('='*25, f)
    # find factor definitions (lines like F['name'] = ...)
    for line in txt.split('\n'):
        if re.match(r"^F\[['\"]", line.strip()) or re.match(r"^# [A-Z]\d+", line.strip()):
            print(line.strip()[:110])
    print()