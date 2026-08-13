"""Trader 2030-10-17 smoke test: verify the 5-factor live computation pipeline
and build_target produce a valid full-investment target from current data."""
import math
import strategy as st
from alphacrafter.sim.utils import get_account_dict

acct = get_account_dict()
assets = list(acct.get("watch_list", []))
print("assets:", len(assets), assets)

live = st._live_factors(assets)
print("live factors computed:", sorted(live.keys()))
for fid, vals in live.items():
    n_fin = sum(1 for v in vals if v == v)
    print(f"  {fid}: finite {n_fin}/15")
    for a, v in zip(assets, vals):
        if v == v:
            print(f"     {a}: {v:.4f}")

# Build a target using the persisted ensemble and current date state
import json
from datetime import date as _date
date_state = json.load(open(st.DATE_PATH))
cur = date_state["current_date"]
td = date_state["trading_days"]
weekdays = [x for x in td if _date.fromisoformat(x).weekday() < 5]
k = weekdays.index(cur) - weekdays.index(st.ONLINE_START)
print("block k mod 10:", k % 10)

ens = st._load_ensemble()
print("ensemble:", [(e["factor_id"], round(e["weight"], 4), e["direction"]) for e in ens])

cur_w = st._current_weights(acct, assets)
built = st.build_target(assets, date_state, ens, current_weights=cur_w)
if built is None:
    print("build_target returned None")
else:
    w, fc, used, meta = built
    print("used factors:", used)
    print("sum weights:", round(sum(w.values()), 6))
    tot = sum(w.values())
    ok = math.isfinite(tot) and abs(tot - 1.0) < 1e-6 and all(math.isfinite(w[a]) and w[a] >= 0 for a in assets)
    print("target valid:", ok)
    print("meta risk/vix/m20/disp:", round(meta["risk"], 3), round(meta["vix"], 2), round(meta["m20"], 4), round(meta["disp"], 4))
    for a in sorted(w, key=lambda x: -w[x]):
        print(f"  {a}: w={w[a]:.4f}  r20={meta['r20'][a]*100:.1f}%  cap={meta['cap_map'].get(a, 0.14)}")
