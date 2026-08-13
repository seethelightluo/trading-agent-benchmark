"""Trader pre-step diagnostic: regime + proposed target vs current (no account mutation)."""
import json
import sys
sys.path.insert(0, ".")
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
import strategy as S

acc = get_account_dict()
assets = list(acc.get("watch_list", []))
date_state = json.loads(Path("../persistent/date.json").read_text())
print("current_date:", date_state.get("current_date"), "visible:", date_state.get("visible_through"))

# regime signals
try:
    vf = get_index_daily_data("VIX", days=40)
    print("VIX last:", float(vf["close"].iloc[-1]), "5d ago:", float(vf["close"].iloc[-6]) if len(vf) > 5 else None)
except Exception as e:
    print("VIX err", e)

closes = S._closes(assets)
risk, vix, m20, disp = S._regime(closes, assets)
print(f"regime risk={risk:.3f} vix={vix:.1f} m20={m20*100:.2f}% disp20={disp*100:.2f}%")

r20 = {}
for a in assets:
    c = closes.get(a)
    r20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0) if (c is not None and len(c) >= 21) else float("nan")
print("20d returns:")
for a in assets:
    print(f"  {a:10s} {r20[a]*100:8.2f}%")

cur_w = S._current_weights(acc, assets)
print("\ncurrent weights:")
for a in assets:
    print(f"  {a:10s} {cur_w[a]*100:6.2f}%")

# proposed target
ensemble = S._load_ensemble()
print("\nensemble:", [f["factor_id"] for f in ensemble])
built = S.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target returned None")
else:
    w, fc, used, meta = built
    print("used factors:", used)
    print("meta:", {k: (round(v,3) if isinstance(v, float) else v) for k, v in meta.items() if k in ("risk","vix","m20","disp","n_factors","cap_map")})
    print("proposed weights:")
    for a in assets:
        print(f"  {a:10s} {w[a]*100:6.2f}%  (cur {cur_w[a]*100:6.2f}%)")
    tot = sum(w.values())
    print("sum weights:", round(tot, 6))
    turn = sum(abs(w[a] - cur_w[a]) for a in assets)
    print("one-way turnover vs current:", round(turn, 4))
    edge = sum(fc[a] * (w[a] - cur_w[a]) for a in assets)
    print("gross edge (fc*(w-cur)):", round(edge, 6), " 3bp*turn:", round(0.0003 * turn, 6))
