"""Trader regime snapshot for 2030-03-26 cycle (post safety-advance)."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
assets = list(acct.get("watch_list", []))
print("account: total", acct.get("total_assets"), "cash", acct.get("available_cash"))

rows = []
for a in assets:
    df = get_stock_daily_data(a, days=80)
    if df is None or len(df) < 25:
        print(a, "NO DATA")
        continue
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    ret20 = s.iloc[-1] / s.iloc[-21] - 1
    ret60 = (s.iloc[-1] / s.iloc[-61] - 1) if len(s) >= 61 else float("nan")
    ma20 = s.rolling(20).mean().iloc[-1]
    above = s.iloc[-1] >= ma20
    rows.append((a, round(float(ret20) * 100, 2), round(float(ret60) * 100, 2), bool(above)))

print(f"{'asset':10s} {'ret20%':>8s} {'ret60%':>8s} {'>MA20':>6s}")
for a, r20, r60, ab in rows:
    print(f"{a:10s} {r20:8.2f} {r60:8.2f} {str(ab):>6s}")

for obs in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    df = get_index_daily_data(obs, days=80)
    if df is None or len(df) < 25:
        print(obs, "NO DATA")
        continue
    s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
    print(obs, "last", round(float(s.iloc[-1]), 3), "20d%", round(float(s.iloc[-1] / s.iloc[-21] - 1) * 100, 2))