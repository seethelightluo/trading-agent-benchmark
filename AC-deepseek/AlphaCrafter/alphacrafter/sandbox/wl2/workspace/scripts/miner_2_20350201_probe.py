"""miner_2 2035-02-01 data probe: visible range, recent regime, macro state."""
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
import pandas as pd
import numpy as np
import json

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
print("watch_list:", ASSETS)

date_state = json.load(open("../persistent/date.json"))
VISIBLE = date_state["visible_through"]
print("current_date:", date_state["current_date"], "visible_through:", VISIBLE)

series = {}
for s in ASSETS:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None:
        print(f"{s}: NO DATA")
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[df.index <= pd.Timestamp(VISIBLE)]
    series[s] = df["close"].astype(float)
    print(f"{s}: rows={len(df)} last={df.index[-1].date()} last_close={df['close'].iloc[-1]:.2f}")

panel = pd.DataFrame(series)
ret = panel.pct_change()
print("\npanel dates:", panel.index[0].date(), "->", panel.index[-1].date(), "rows:", len(panel))

def r(days):
    return (panel.iloc[-1] / panel.iloc[-1 - days] - 1.0) * 100

for days in [5, 10, 20, 60, 180, 252]:
    rr = r(days).sort_values(ascending=False)
    print(f"\n=== {days}d return (%) ===")
    print(rr.round(2).to_string())

print("\n=== 20d volatility (annualized %) ===")
print((ret.tail(20).std() * np.sqrt(252) * 100).round(1).sort_values(ascending=False).to_string())

# macro
for m in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{m}.csv', parse_dates=['date'])
    df = df.set_index('date').sort_index()
    df = df[df.index <= pd.Timestamp(VISIBLE)]
    print(f"\n{m}: last={df.index[-1].date()} last={df['close'].iloc[-1]:.2f}")
    for d in [5, 20, 60]:
        if len(df) > d:
            print(f"  {d}d chg %: {((df['close'].iloc[-1]/df['close'].iloc[-1-d]-1)*100):.2f}")

# 252d range position snapshot
hp = panel.rolling(252, min_periods=60).max()
lp = panel.rolling(252, min_periods=60).min()
print("\n=== 252d range position (0=low,1=high) ===")
print(((panel.iloc[-1] - lp.iloc[-1]) / (hp.iloc[-1] - lp.iloc[-1])).round(3).sort_values(ascending=False).to_string())
