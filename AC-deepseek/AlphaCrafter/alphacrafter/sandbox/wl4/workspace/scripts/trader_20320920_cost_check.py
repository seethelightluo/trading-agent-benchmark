"""Determine whether the 09-20 proposal executed: compare position cost prices to 09-20 closes."""
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
positions = {p["symbol"]: p for p in acct.get("positions", [])}
watch = acct.get("watch_list", [])

rows = []
for sym in watch:
    df = None
    try:
        df = get_stock_daily_data(symbol=sym, days=30)
    except Exception:
        df = None
    if df is None or len(df) < 2:
        try:
            df = get_index_daily_data(symbol=sym, days=30)
        except Exception:
            df = None
    if df is None or len(df) < 2:
        rows.append((sym, None, None, None, None))
        continue
    df = df.sort_values("date").reset_index(drop=True)
    # find 2032-09-20 and last close
    d20 = df[df["date"] == pd.Timestamp("2032-09-20")]
    last = df.iloc[-1]
    c20 = float(d20.iloc[0]["close"]) if len(d20) else None
    clast = float(last["close"])
    dlast = last["date"]
    p = positions.get(sym)
    cost = float(p["cost_price"]) if p else None
    rows.append((sym, c20, cost, clast, dlast))

print(f"{'sym':10s} {'close09-20':>12s} {'cost':>12s} {'close_last':>12s} {'last_date':>12s}  cost==c20?")
for sym, c20, cost, clast, dlast in rows:
    eq = "YES" if (c20 is not None and cost is not None and abs(c20 - cost) / c20 < 1e-6) else "no"
    print(f"{sym:10s} {str(c20):>12s} {str(cost):>12s} {str(clast):>12s} {str(dlast):>12s}  {eq}")
