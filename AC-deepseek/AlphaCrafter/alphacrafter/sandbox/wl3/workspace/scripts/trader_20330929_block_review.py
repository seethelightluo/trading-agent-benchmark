import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

OBS = {"DXY","VIX","USDCNY","USDJPY","EURUSD"}
assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def get_df(sym):
    try:
        if sym in OBS:
            return get_index_daily_data(sym, days=30)
        return get_stock_daily_data(sym, days=30)
    except Exception:
        return None

# block 09-15..09-29: find close on 2033-09-14 (prev day) and 2033-09-29
rows = []
for a in assets:
    df = get_df(a)
    if df is None or len(df) < 5:
        print(a, "no data"); continue
    df = df.sort_values("date").reset_index(drop=True)
    dts = pd.Series([str(x)[:10] for x in df["date"]])
    last = df.iloc[-1]
    sub = df[dts <= "2033-09-14"]
    if len(sub) == 0:
        print(a, "no prev"); continue
    prev = sub.iloc[-1]
    r = last["close"]/prev["close"] - 1.0
    rows.append((a, r*100, prev["close"], last["close"]))

for a, r, p0, p1 in sorted(rows, key=lambda x: -x[1]):
    print(f"{a:10s} block ret {r:+.2f}%  ({p0:.2f} -> {p1:.2f})")
print()
tot = sum(r for _,r,_,_ in rows)/len(rows)
print(f"mean asset ret: {tot:+.2f}%")
