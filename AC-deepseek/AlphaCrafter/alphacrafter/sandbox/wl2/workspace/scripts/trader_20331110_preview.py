"""Preview the strategy's target for block start 2033-11-10 (read-only, no mutation)."""
import json
import sys
sys.path.insert(0, ".")
import strategy as st
from alphacrafter.sim.utils import get_account_dict

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
date_state = json.load(open("../persistent/date.json"))
ensemble = st._load_ensemble()
print("ensemble:", [(f["factor_id"], round(f["weight"], 4), f["direction"]) for f in ensemble])

cur_w = st._current_weights(acc, assets)
built = st.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target returned None")
    raise SystemExit
weights, forecast, used, meta = built
print("used factors:", used)
print("risk=%.3f vix=%.1f m20=%.4f disp20=%.4f" % (meta["risk"], meta["vix"], meta["m20"], meta["disp"]))
print("cap_map:", meta["cap_map"])
print("\n=== target weights vs current ===")
tot_turn = 0.0
for a in assets:
    cw = cur_w.get(a, 0.0)
    tw = weights[a]
    tot_turn += abs(tw - cw)
    flag = " <-- cap" if a in (meta.get("cap_map") or {}) else ""
    print(f"{a:10s} cur={cw*100:6.2f}%  tgt={tw*100:6.2f}%  chg={ (tw-cw)*100:+6.2f}pp  fcst={forecast[a]*100:+6.2f}%{flag}")
print("\nsum weights:", round(sum(weights.values()), 6), "| one-way turnover:", round(tot_turn * 100, 2), "%")
