"""Trader cycle-74 diagnostic: current target preview + recent returns at 2030-05-16."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, ".")
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
import strategy as S

# date state
date_state = json.loads(Path("persistent/date.json").read_text())
print("current_date:", date_state.get("current_date"))
print("visible_through:", date_state.get("visible_through"))
td = date_state.get("trading_days", [])
print("n trading days:", len(td), "last 3:", td[-3:])

account = get_account_dict()
assets = list(account.get("watch_list", []))
print("assets:", len(assets))
print("net_assets:", account.get("net_assets"), "cash:", account.get("available_cash"))
for p in account.get("positions", []):
    print(f"  pos {p['symbol']}: qty {p.get('quantity'):.4f} mv {p.get('market_value',0):,.0f} px {p.get('current_price',0):.4f}")

# recent returns
print("\n--- 20d / 60d returns (visible) ---")
for a in assets:
    df = get_stock_daily_data(a, days=130)
    if df is None or len(df) < 62:
        print(f"  {a}: no data"); continue
    c = df["close"].astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1.0
    r60 = c.iloc[-1] / c.iloc[-61] - 1.0
    print(f"  {a:10s} r20 {r20*100:7.2f}%  r60 {r60*100:7.2f}%  px {c.iloc[-1]:.2f}")

# VIX / macro
try:
    vf = get_index_daily_data("VIX", days=40)
    print("\nVIX last:", float(vf["close"].iloc[-1]) if vf is not None else "n/a")
except Exception as e:
    print("vix err", e)

# build current target
ensemble = S._load_ensemble()
print("\nensemble factors:", len(ensemble))
built = S.build_target(assets, date_state, ensemble, current_weights=S._current_weights(account, assets))
if built is None:
    print("build_target returned None")
else:
    weights, forecast, used, meta = built
    print("used factors:", used)
    print(f"risk={meta['risk']:.3f} vix={meta['vix']:.1f} m20={meta['m20']*100:.2f}% disp={meta['disp']*100:.3f}")
    print("cap_map:", {k: v for k, v in meta['cap_map'].items()})
    print("\n--- proposed target ---")
    for a in sorted(weights, key=lambda x: -weights[x]):
        print(f"  {a:10s} w {weights[a]*100:6.2f}%  z {meta['z'][a]:+6.2f}  r20 {meta['r20'][a]*100:7.2f}%  fcst {forecast[a]*100:+6.2f}%")
    print("sum:", sum(weights.values()))
    cur = S._current_weights(account, assets)
    turn = sum(abs(weights[a] - cur[a]) for a in assets)
    print("one-way turnover vs current:", turn)
