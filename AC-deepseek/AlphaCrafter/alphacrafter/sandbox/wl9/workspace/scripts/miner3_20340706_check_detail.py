"""Check factor details for a few key factors"""
import json

for fname in ["factors/beta_VIX_60.json", "factors/mom_120d_skip5.json", "factors/bb_width_20d.json", "factors/kaufman_eff_20d.json"]:
    d = json.load(open(fname))
    print(f"===== {d.get('factor_id','?')} =====")
    metrics = d.get("validation",{}).get("metrics",{})
    print(f"status: {d.get('validation',{}).get('status','?')}")
    print(f"IC: {metrics.get('IC','?')}")
    print(f"ICIR: {metrics.get('ICIR','?')}")
    print(f"turnover: {metrics.get('turnover','?')}")
    print(f"coverage: {metrics.get('coverage','?')}")
    print(f"last_validated: {d.get('last_validated','?')}")
    print(f"calculation desc: {d.get('calculation',{}).get('description','?')}")
    print()