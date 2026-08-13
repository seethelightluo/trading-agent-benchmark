import json, os
# miner_1 scripts
m1 = [f for f in sorted(os.listdir('scripts')) if f.startswith('miner_1')]
print("miner_1 scripts:", len(m1))
for f in m1[-12:]:
    print(f)
print()
# check latest results json from miner_3
p = 'scripts/miner_3_20310918_results_batch34.json'
if os.path.exists(p):
    d = json.load(open(p))
    print(json.dumps(d, default=str, indent=1)[:3000])