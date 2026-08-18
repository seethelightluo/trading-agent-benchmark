"""Trader pre-step inspection: current regime, VIX, factor scores, target weights."""
import json
from pathlib import Path
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
import pandas as pd

acct = get_account_dict()
assets = acct.get("watch_list", [])
print("date-ish account:", {k: acct.get(k) for k in ("total_assets","net_assets","available_cash","gross_position_rate")})
print("positions:", [(p["symbol"], round(p.get("quantity",0),2), round(p.get("market_value",0),0)) for p in acct.get("positions",[])])
print("pending orders:", len(acct.get("orders",[])))

frames = {}
closes = {}
for a in assets:
    f = get_stock_daily_data(a, days=300)
    frames[a] = f
    closes[a] = f["close"].astype(float) if f is not None and "close" in f else None

panel = pd.concat([c.rename(a) for a,c in closes.items() if c is not None and len(c)>=140], axis=1, join="inner")
rets = panel.pct_change().dropna()
last = panel.iloc[-1]
print("\nlast closes:", {a: round(float(last[a]),2) for a in panel.columns})
print("20d rets:", {a: round(float(panel[a].iloc[-1]/panel[a].iloc[-21]-1)*100,1) for a in panel.columns})
print("60d rets:", {a: round(float(panel[a].iloc[-1]/panel[a].iloc[-61]-1)*100,1) for a in panel.columns})

mkt = rets.mean(axis=1)
r20 = float(mkt.tail(20).mean()); v20 = float(mkt.tail(20).std())
trend = r20/v20*(20**0.5) if v20 and v20>1e-12 else 0.0
regime = "bull" if trend>1.0 else ("bear" if trend<-1.0 else "sideways")
print(f"\nregime={regime} trend_t={trend:.3f} r20={r20:.5f}")

vf = get_index_daily_data("VIX", days=300)
if vf is not None:
    vix = vf["close"].astype(float)
    print("VIX last:", round(float(vix.iloc[-1]),2), "20d ago:", round(float(vix.iloc[-21]),2), "60d ago:", round(float(vix.iloc[-61]),2), "max60:", round(float(vix.tail(60).max()),2))
for m in ("DXY","USDCNY","USDJPY","EURUSD"):
    mi = get_index_daily_data(m, days=60)
    if mi is not None:
        c = mi["close"].astype(float)
        print(f"{m} last {round(float(c.iloc[-1]),3)} 20d {round(float(c.iloc[-1]/c.iloc[-21]-1)*100,2)}%")
# dump latest ensemble info
ens = json.loads(Path("factor_ensemble.json").read_text())
print("\nensemble generated_at:", ens.get("generated_at"), "n_factors:", len(ens.get("selected_factors",[])))