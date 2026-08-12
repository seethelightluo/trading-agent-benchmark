import json
r = json.load(open('scripts/miner_3_20280810_results_batch2.json'))
print("candidates:", list(r.keys()))
print()
for k, v in r.items():
    ic = v.get('ic'); icir = v.get('icir')
    gate = "PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else "fail"
    print(f"{k:28s} ic={ic:+.4f} icir={icir:+.4f} hit={v.get('hit'):.3f} turn={v.get('turn'):.2f} cov={v.get('cov'):.3f} ge8={v.get('ge8'):.3f} rho={v.get('rho')} [{gate}]")