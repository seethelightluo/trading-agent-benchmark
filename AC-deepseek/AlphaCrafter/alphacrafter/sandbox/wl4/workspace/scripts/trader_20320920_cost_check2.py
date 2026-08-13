"""Verify cost prices == 09-03 closes (09-06 executed proposal) to confirm 09-20 skip."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
positions = {p["symbol"]: p for p in acct.get("positions", [])}
watch = acct.get("watch_list", [])

for sym in watch:
    df = None
    try:
        df = get_stock_daily_data(symbol=sym, days=40)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(symbol=sym, days=40)
        except Exception:
            df = None
    if df is None or len(df) < 2:
        print(f"{sym:10s} no data")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    d03 = df[df["date"] == pd.Timestamp("2032-09-03")]
    c03 = float(d03.iloc[0]["close"]) if len(d03) else None
    cost = float(positions[sym]["cost_price"])
    eq = "YES" if (c03 is not None and abs(c03 - cost) / c03 < 1e-6) else "no"
    print(f"{sym:10s} close09-03={str(c03):>14s} cost={cost:14.6f}  cost==c03? {eq}")
