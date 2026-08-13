"""Check whether cost prices == 09-17 closes (would indicate 09-20 proposal EXECUTED at prev close)."""
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
    d17 = df[df["date"] == pd.Timestamp("2032-09-17")]
    c17 = float(d17.iloc[0]["close"]) if len(d17) else None
    cost = float(positions[sym]["cost_price"])
    eq = "YES" if (c17 is not None and abs(c17 - cost) / c17 < 1e-6) else "no"
    print(f"{sym:10s} close09-17={str(c17):>14s} cost={cost:14.6f}  cost==c17? {eq}")
