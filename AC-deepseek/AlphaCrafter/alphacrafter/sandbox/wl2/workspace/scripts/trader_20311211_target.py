import json
import sys
sys.path.insert(0, ".")
from pathlib import Path
import strategy as st

d = json.load(open("../persistent/date.json"))
td = d["trading_days"]
date_state = {"current_date": "2031-12-11", "visible_through": "2031-12-10", "trading_days": td}
assets = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

ensemble = st._load_ensemble()
print("ensemble factors:", [(f["factor_id"], round(f["weight"], 4), f["direction"]) for f in ensemble])

built = st.build_target(assets, date_state, ensemble, current_weights=None)
if built is None:
    print("build_target returned None")
    sys.exit(0)
weights, forecast, used, meta = built
print("used:", used)
print("risk %.3f vix %.1f m20 %.4f disp %.4f" % (meta["risk"], meta["vix"], meta["m20"], meta["disp"]))
print("---target weights (sorted desc)---")
for a, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"{a:10s} {w*100:6.2f}%  r20={meta['r20'].get(a,0)*100:7.2f}%  cap={meta['cap_map'].get(a, st.CAP):.2f}")
print("sum:", sum(weights.values()))
print("---forecast top/bottom---")
for a, f in sorted(forecast.items(), key=lambda x: -x[1])[:5]:
    print("top", a, round(f, 4))
for a, f in sorted(forecast.items(), key=lambda x: x[1])[:5]:
    print("bot", a, round(f, 4))
