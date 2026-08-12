import json, os
for f in ['scripts/miner3_20300524_screenA_results.json']:
    if os.path.exists(f):
        d = json.load(open(f))
        print(f, "keys:", list(d.keys()))
        for k, v in d.items():
            ic1 = v['h']['1']
            print(f"{k:24s} ic1={ic1['ic']:.4f} icir1={ic1['icir']:.3f} hit={ic1['hit']:.2f} n={ic1['n']} turn={v['turn']:.3f}")
