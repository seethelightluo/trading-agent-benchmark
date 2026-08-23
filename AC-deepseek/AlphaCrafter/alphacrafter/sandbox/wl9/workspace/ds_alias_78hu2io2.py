import json
for f in ['factors/evicted/bbz_20d.json.reason.json','factors/evicted/dside_ratio_21.json.reason.json']:
    print('==',f)
    print(open(f).read()[:800])