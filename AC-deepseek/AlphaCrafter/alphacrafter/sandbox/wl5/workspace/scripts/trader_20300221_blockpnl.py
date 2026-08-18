import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 'XAU',
         'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def get_close(sym):
    df = get_stock_daily_data(sym, days=30)
    if df is None or len(df) < 15:
        df = get_index_daily_data(sym, days=30)
    if df is None:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df[['date', 'close']].set_index('date')['close'].astype(float)

acc = get_account_dict()
total_end = acc['total_assets']
pos = {p['symbol']: p for p in acc['positions']}

print(f"{'asset':10s} {'px0207':>10s} {'px0306':>10s} {'blk_ret':>8s} {'end_val':>10s} {'start_val':>10s} {'contrib_pp':>10s}")
rows = []
for a in WATCH:
    c = get_close(a)
    if c is None or len(c) < 12:
        print(a, "no data")
        continue
    px_prev = float(c.iloc[-11])   # close on 2027-02-20 (10 trading days before last)
    px_last = float(c.iloc[-1])    # close on last day 2030-03-06
    # safer: find the close 10 trading days before the last available
    end_val = pos[a]['market_value'] if a in pos else 0.0
    r = px_last / px_prev - 1.0
    start_val = end_val / (1.0 + r) if r > -0.999 else 0.0
    contrib = start_val / total_end * r * 100.0  # approx pp of total return
    rows.append((a, px_prev, px_last, r, end_val, start_val, contrib))
    print(f"{a:10s} {px_prev:10.2f} {px_last:10.2f} {r*100:7.2f}% {end_val:10.2f} {start_val:10.2f} {contrib:9.3f}pp")

print("\nsum contrib pp:", round(sum(x[6] for x in rows), 3))
print("period return pct:", round((total_end / 1134264.0 - 1.0) * 100.0, 3))
