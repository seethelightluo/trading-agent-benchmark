"""Trader attribution for block 2031-09-18 -> 2031-10-02."""
import json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

acc = json.load(open('../persistent/account.json'))
positions = {p['symbol']: p for p in acc['positions']}
total_end = acc['total_assets']

SYMS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
        'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def get_df(sym):
    df = get_stock_daily_data(symbol=sym, days=30)
    if df is None or len(df) == 0:
        df = get_index_daily_data(symbol=sym, days=30)
    return df

rows = []
for sym in SYMS:
    df = get_df(sym)
    if df is None or len(df) < 15:
        print('NO DATA', sym)
        continue
    df = df.sort_values('date')
    # block: last completed day before 09-18 is 09-17; last day of block is 10-01/10-02
    p_start = df.iloc[-11]['close']   # 09-17 close
    p_end = df.iloc[-1]['close']      # 10-01/10-02 close
    ret = p_end / p_start - 1.0
    mv_end = positions[sym]['market_value']
    mv_start = mv_end / (1 + ret) if (1 + ret) != 0 else mv_end
    w_start = mv_start / (mv_start * 0 + total_end)  # placeholder
    # use share of total using start mv estimated
    rows.append((sym, ret, mv_end, mv_start))

sum_mv_start = sum(r[2] for r in rows)
contrib_total = 0.0
print(f"{'SYM':10s} {'ret%':>8s} {'mv_start':>12s} {'w_start%':>8s} {'contrib_pp':>10s}")
for sym, ret, mv_end, mv_start in sorted(rows, key=lambda r: -r[1]):
    w = mv_start / sum_mv_start
    c = w * ret * 100
    contrib_total += c
    print(f"{sym:10s} {ret*100:8.2f} {mv_start:12.2f} {w*100:8.2f} {c:10.4f}")
print('SUM contrib pp:', round(contrib_total, 4))
print('realized period return: 0.77%')
