import json
with open('scripts/miner_3_20290208_revalidate_results.json') as f:
    d = json.load(f)
# print summary of all factors: ok status, ic, icir, last250
print(f"{'factor':<28}{'ok':<6}{'IC':<10}{'ICIR':<9}{'last250IC':<10}{'last250ICIR':<10}{'corr':<8}")
for k, v in d.items():
    ok = v.get('ok')
    print(f"{k:<28}{str(ok):<6}{v.get('ic',0):<10.5f}{v.get('icir',0):<9.4f}{v.get('regime',{}).get('last250',{}).get('ic',0):<10.5f}{v.get('regime',{}).get('last250',{}).get('icir',0):<10.4f}{v.get('max_abs_library_correlation',0):<8.3f}")