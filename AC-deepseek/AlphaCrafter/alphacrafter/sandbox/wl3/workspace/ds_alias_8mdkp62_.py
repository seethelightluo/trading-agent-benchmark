import json
d = json.load(open('scripts/miner_3_20310918_results_batch34.json'))
for k, v in d.items():
    print(f"{k:24s} PASS={v.get('PASS')} ic={v.get('ic'):+.4f} icir={v.get('icir'):+.3f} rho={v.get('rho'):.3f} rho_id={v.get('rho_id')} ic_recent={v.get('ic_recent'):+.4f}")