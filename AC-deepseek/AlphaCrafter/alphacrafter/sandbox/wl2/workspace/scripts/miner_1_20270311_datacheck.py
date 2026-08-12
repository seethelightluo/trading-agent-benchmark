"""miner_1 datacheck 2027-03-11 cycle: verify data state through visible_through."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

date_state = json.load(open("../persistent/date.json"))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
print("current_date:", date_state["current_date"])
print("visible_through:", VISIBLE)
print("total trading days:", len(TRADING_DAYS))

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
print("watch_list:", ASSETS)

for s in ASSETS:
    df = get_stock_daily_data(s, days=2500)
    if df is None:
        print(s, "NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    last = df.index[-1]
    close = df["close"].astype(float)
    # flat check over last 60 rows
    tail = close.tail(60)
    flat = float((tail.diff().abs() < 1e-12).mean())
    last5 = close.tail(5).round(4).tolist()
    print("%-12s rows=%4d last=%s flat60=%.2f last5=%s" % (s, len(df), last, flat, last5))
