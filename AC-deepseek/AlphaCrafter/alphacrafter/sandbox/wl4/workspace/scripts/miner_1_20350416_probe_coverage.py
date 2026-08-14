"""miner_1 probe: data coverage through visible_through=2035-04-13."""
import pandas as pd
import json

VIS = '2035-04-13'
CAL = pd.Index(json.load(open('../persistent/date.json'))['trading_days'])
CAL = CAL[CAL <= VIS]
print('calendar days through', VIS, ':', len(CAL))

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

rows = []
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df[df['date'] <= VIS].set_index('date').reindex(CAL)
    n = df['close'].notna().sum()
    # dense blocks (consecutive non-nan runs) summary
    mask = df['close'].notna()
    runs, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    rows.append((s, n, len(runs), max(runs) if runs else 0, runs[-1] if runs else 0))

print(f"{'asset':10s} {'n_obs':>6s} {'runs':>5s} {'max_run':>8s} {'last_run':>8s}")
for r in rows:
    print(f"{r[0]:10s} {r[1]:6d} {r[2]:5d} {r[3]:8d} {r[4]:8d}")
