from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

def get(sym):
    df = get_index_daily_data(symbol=sym, days=60)
    if df is None or len(df) == 0:
        df = get_stock_daily_data(symbol=sym, days=60)
    return df

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
print("=== Per-asset closes and block returns ===")
rows = []
for a in assets:
    df = get(a)
    if df is None or len(df) < 5:
        print(a, "NO DATA")
        continue
    df = df.sort_values('date').reset_index(drop=True)
    last = df.iloc[-1]
    d_last = last['date']
    sub = df[df['date'] <= pd.Timestamp('2034-11-26')]
    sub2 = df[df['date'] <= pd.Timestamp('2034-11-27')]
    if len(sub) == 0 or len(sub2) == 0:
        print(a, "insufficient window", df['date'].min(), df['date'].max())
        continue
    c_1126 = sub.iloc[-1]['close']
    c_1127 = sub2.iloc[-1]['close']
    c_end = last['close']
    r_cost = (c_end - c_1126)/c_1126      # return from cost date (11-26) to end
    r_block = (c_end - c_1127)/c_1127     # return from proposal day to end
    rows.append((a, c_1126, c_1127, c_end, r_cost, r_block, d_last))
    print(f"{a:10s} c1126={c_1126:12.4f} c1127={c_1127:12.4f} c_end={c_end:12.4f} r_cost={r_cost:8.3%} r_block={r_block:8.3%} last={d_last}")

print("\n=== sorted by r_cost ===")
for r in sorted(rows, key=lambda x: x[4]):
    print(f"{r[0]:10s} r_cost={r[4]:8.3%}  r_block={r[5]:8.3%}")
