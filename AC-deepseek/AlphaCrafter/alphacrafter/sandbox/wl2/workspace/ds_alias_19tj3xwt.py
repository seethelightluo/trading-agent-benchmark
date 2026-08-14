import json
d=json.load(open('scripts/miner_3_20340803_revalidate_results.json'))
oks = {k:v for k,v in d.items() if v.get('ok')}
print("PASSED factors:", list(oks.keys()))
for k,v in oks.items():
    print(k, "ic=",round(v['ic'],4), "icir=",round(v['icir'],4), "last250_ic=",v['regime'].get('last250',{}).get('ic'), "maxrho=",v.get('max_abs_library_correlation'))
print("\n--- last250 ic for all library factors ---")
for k,v in sorted(d.items(), key=lambda kv: abs(kv[1].get('regime',{}).get('last250',{}).get('ic',0) or 0), reverse=True)[:15]:
    l250 = v.get('regime',{}).get('last250',{})
    print(f"{k:26s} full_ic={v['ic']:+.4f} full_icir={v['icir']:+.4f} l250_ic={l250.get('ic','NA')} l250_icir={l250.get('icir','NA')} n={l250.get('n','NA')}")