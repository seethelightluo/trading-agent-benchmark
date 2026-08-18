"""Trader regime check at 2030-03-12 decision. Data visible thru 2030-03-11."""
from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
)
import pandas as pd

acct = get_account_dict()
print("date keys:", [k for k in acct.keys()][:12])
print("total_assets:", acct.get("total_assets"))
print("watch_list:", acct.get("watch_list"))

watch = acct.get("watch_list", [])
rows = {}
for a in watch:
    df = get_stock_daily_data(a, days=80)
    if df is None or len(df) < 30:
        print(a, "no data")
        continue
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    r = s.pct_change()
    ma20 = s.rolling(20).mean().iloc[-1]
    last = s.iloc[-1]
    ret20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else float("nan")
    ret5 = s.iloc[-1] / s.iloc[-6] - 1 if len(s) > 6 else float("nan")
    vol20 = r.tail(20).std()
    rows[a] = dict(last=round(float(last), 2), ma20=round(float(ma20), 2),
                   above_ma20=bool(last > ma20), ret20=round(float(ret20), 4),
                   ret5=round(float(ret5), 4), vol20=round(float(vol20), 4))
    print(f"{a:10s} last={last:10.2f} ma20={ma20:10.2f} aboveMA20={last>ma20!s:5s} "
          f"r20={ret20:+.2%} r5={ret5:+.2%} vol20={vol20:.2%}")

for a in ["DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"]:
    df = get_index_daily_data(a, days=40)
    if df is None or len(df) < 5:
        print(a, "no data")
        continue
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    r20 = s.iloc[-1] / s.iloc[-21] - 1 if len(s) > 21 else float("nan")
    print(f"{a:8s} last={s.iloc[-1]:10.2f} r20={r20:+.2%}")

# positions snapshot
print("\npositions:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']:10s} qty={p['quantity']:12.4f} mktval={p['market_value']:12.2f} plr={p.get('profit_loss_rate',0):+.2%}")
print("cash:", acct.get("available_cash"), "gross pos rate:", acct.get("gross_position_rate"))
