"""Trader preflight: show the live factor values, target weights and meta
the strategy would build for the 2032-10-28 block start (data through 10-27).
Read-only: does not modify account/date."""
import json
import sys
sys.path.insert(0, ".")
import strategy as S

assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

date_state = json.load(open("../persistent/date.json"))
ensemble = S._load_ensemble()
print("ensemble loaded:", [(f["factor_id"], round(f["weight"], 4), f["direction"]) for f in ensemble])

# current weights from account
acc = json.load(open("../persistent/account.json"))
cur_w = S._current_weights(acc, assets)
print("\ncurrent weights:")
for a in assets:
    print(f"  {a:10s} {cur_w[a]*100:6.2f}%")

built = S.build_target(assets, date_state, ensemble, current_weights=cur_w)
if built is None:
    print("build_target returned None!")
    sys.exit(1)
weights, forecast, used, meta = built
print("\nused factors:", used)
print("meta: risk=%.3f vix=%.1f m20=%.4f disp=%.4f n_factors=%d" %
      (meta["risk"], meta["vix"], meta["m20"], meta["disp"], meta["n_factors"]))
print("\nTARGET weights (sum=%.6f):" % sum(weights.values()))
for a in assets:
    print(f"  {a:10s} {weights[a]*100:6.2f}%   (cur {cur_w[a]*100:6.2f}%)  fcast {forecast[a]*100:+.2f}%  r20 {meta['r20'][a]*100:+.2f}%  cap {meta['cap_map'].get(a, S.CAP)}")
turn = sum(abs(weights[a] - cur_w[a]) for a in assets)
print("\nOne-way turnover vs current: %.2f%%" % (turn * 100))

# recent regime snapshot: 20d/60d returns per asset
from alphacrafter.sim.utils import get_stock_daily_data
print("\nregime snapshot (visible 10-27):")
for a in assets:
    df = get_stock_daily_data(a, days=90)
    if df is None or len(df) < 65:
        print(f"  {a:10s} no data")
        continue
    c = df["close"].astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1
    r60 = c.iloc[-1] / c.iloc[-61] - 1
    print(f"  {a:10s} r20 {r20*100:+7.2f}%  r60 {r60*100:+7.2f}%  last {c.iloc[-1]:.4f}")
