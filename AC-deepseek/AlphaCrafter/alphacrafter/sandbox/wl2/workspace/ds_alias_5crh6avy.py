import json
d = json.load(open('scripts/miner_3_20270923_revalidate_results.json'))
ok_fids = [k for k,v in d.items() if v.get('ok')]
print('OK factors as of 2027-09-23:', ok_fids)
# also check 20270630
d2 = json.load(open('scripts/miner_3_20270630_revalidate_results.json'))
ok2 = [k for k,v in d2.items() if v.get('ok')]
print('OK factors as of 2027-06-30:', ok2)
