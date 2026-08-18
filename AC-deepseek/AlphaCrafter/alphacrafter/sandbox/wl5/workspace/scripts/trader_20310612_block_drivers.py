"""Compute 2031-05-29 -> 2031-06-12 block drivers from account positions."""
import json

before = {
 "000300.SH": 15.9916, "SPX": 7.9151, "HSI": 3.0777, "N225": 2.0079, "SX5E": 12.194,
 "000688.SH": 40.6824, "SOX": 10.7226, "NDX": 2.4409, "XAU": 17.0889, "COPPER": 16408.9933,
 "WTI": 617.9911, "BTC": 1.3243, "ETH": 49.2843, "US10Y": 18460.7935, "CN10Y": 43089.654,
}
acc = json.load(open('../persistent/account.json'))
start_total = 1139875.4464
now = {p['symbol']: p for p in acc['positions']}
# get start prices from cost basis is not right; use account history unavailable.
# Instead fetch daily data for each symbol around 2031-05-28 and 2031-06-11.
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

def get_df(sym):
    df = get_stock_daily_data(symbol=sym, days=40)
    if df is None:
        df = get_index_daily_data(symbol=sym, days=40)
    return df

start_px, end_px = {}, {}
for sym in before:
    df = get_df(sym)
    if df is None or len(df) < 20:
        print("NO DATA", sym); continue
    df = df.sort_values('date')
    # find close on/before 2031-05-28 and last close
    d0 = None
    for _, r in df.iterrows():
        if str(r['date'])[:10] <= '2031-05-28':
            d0 = r['close']
    d1 = df.iloc[-1]['close']
    start_px[sym] = d0
    end_px[sym] = d1
    print(sym, 'start', d0, 'end', d1, 'ret%', round((d1/d0-1)*100, 2))

print("\n--- block contribution (pp of start net assets) ---")
tot = 0.0
for sym, qty in before.items():
    if sym not in start_px or start_px[sym] is None:
        continue
    ret = end_px[sym] / start_px[sym] - 1
    w = qty * start_px[sym] / start_total
    contrib = w * ret * 100
    tot += contrib
    print(f"{sym:10s} w={w*100:6.2f}% ret={ret*100:7.2f}% contrib={contrib:+6.2f}pp")
print("sum contrib pp:", round(tot, 2))
print("realized block ret:", round((acc['net_assets']/start_total-1)*100, 2))
