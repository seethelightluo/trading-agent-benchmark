
import json
d = json.load(open('factors/rejected/kaufman_eff_60.json'))
print(json.dumps({k: d.get(k) for k in ['factor_id','factor_name','version','last_validated']}, indent=1))
print("metrics:", json.dumps(d.get('validation',{}).get('metrics',{}), indent=1)[:1200])
