"""Trader regime diagnostic at 2028-02-24 (decision for next block)."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

acct = get_account_dict()
print("NET_ASSETS:", acct.get("net_assets"))
print("CASH:", acct.get("available_cash"))
print("POSITIONS:")
for p in acct.get("positions", []):
    print(f"  {p['symbol']:12s} qty={p['quantity']:12.4f} mv={p['market_value']:12.2f} "
          f"pnl={p['profit_loss']:10.2f} ({p['profit_loss_rate']*100:+.2f}%)")
print("WATCH:", acct.get("watch_list"))

assets = acct.get("watch_list", [])
print("\n--- 20d / 60d returns (visible through prev completed day) ---")
rows = []
for a in assets:
    try:
        df = get_stock_daily_data(a, days=200)
    except Exception:
        df = None
    if df is None or len(df) < 65:
        rows.append((a, float("nan"), float("nan"), 0.0))
        continue
    c = df["close"].astype(float)
    r20 = c.iloc[-1] / c.iloc[-21] - 1.0
    r60 = c.iloc[-1] / c.iloc[-61] - 1.0
    s = c.pct_change().dropna().tail(20)
    vol = float(s.std()) if len(s) >= 5 else float("nan")
    rows.append((a, r20, r60, vol))

rows.sort(key=lambda x: x[1] if x[1] == x[1] else -99)
for a, r20, r60, vol in rows:
    print(f"  {a:12s} 20d={r20*100:8.2f}%  60d={r60*100:8.2f}%  vol20={vol*100:5.2f}%")

# macro observation signals
for sig in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]:
    try:
        df = get_index_daily_data(sig, days=70)
        if df is not None and len(df) > 1:
            c = df["close"].astype(float)
            print(f"  {sig:8s} last={c.iloc[-1]:9.3f}  5d={c.iloc[-1]/c.iloc[-6]-1:+.2%}  "
                  f"20d={c.iloc[-1]/c.iloc[-21]-1:+.2%}")
    except Exception as e:
        print(f"  {sig}: err {e}")
