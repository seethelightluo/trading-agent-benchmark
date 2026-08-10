import pandas as pd, numpy as np, os

CUTOFF = '2026-07-15'
data_dir = '../persistent/stock_data'
instr = ['000300.SH','000688.SH','SPX','NDX','SOX','HSI','N225','SX5E','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows = []
for s in instr:
    df = pd.read_csv(os.path.join(data_dir, s + '.csv'))
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
    last = df.iloc[-1]
    closes = df['close']
    ret5 = closes.iloc[-1]/closes.iloc[-6] - 1 if len(closes) > 6 else np.nan
    ret20 = closes.iloc[-1]/closes.iloc[-21] - 1 if len(closes) > 21 else np.nan
    ret60 = closes.iloc[-1]/closes.iloc[-61] - 1 if len(closes) > 61 else np.nan
    ret120 = closes.iloc[-1]/closes.iloc[-121] - 1 if len(closes) > 121 else np.nan
    r = closes.pct_change()
    vol20 = r.iloc[-20:].std()*np.sqrt(252) if len(r) > 20 else np.nan
    ma20 = closes.iloc[-20:].mean()
    ma60 = closes.iloc[-60:].mean() if len(closes) >= 60 else np.nan
    if closes.iloc[-1] > ma20 > ma60:
        trend = 'UP'
    elif closes.iloc[-1] < ma20 < ma60:
        trend = 'DOWN'
    else:
        trend = 'MIX'
    rows.append((s, str(last['date'].date()), round(float(last['close']), 4), round(float(ret5)*100, 2),
                 round(float(ret20)*100, 2), round(float(ret60)*100, 2), round(float(ret120)*100, 2),
                 round(float(vol20)*100, 1), trend))
print('%-10s %-12s %10s %7s %7s %7s %7s %7s %5s' % ('sym', 'date', 'close', 'r5%', 'r20%', 'r60%', 'r120%', 'vol20%', 'trend'))
for r in rows:
    print('%-10s %-12s %10.2f %7.2f %7.2f %7.2f %7.2f %7.2f %5s' % r)

# dispersion
rets = {}
for s in instr:
    df = pd.read_csv(os.path.join(data_dir, s + '.csv'))
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
    rets[s] = df.set_index('date')['close'].pct_change()
R = pd.DataFrame(rets).dropna()
R60 = R.iloc[-60:]
corr = R60.corr()
vals = corr.values[np.triu_indices(len(corr), k=1)]
print('\nAvg pairwise corr (60d): %.3f | median: %.3f' % (np.nanmean(vals), np.nanmedian(vals)))
print('Avg pairwise corr (10d): %.3f' % np.nanmean(R.iloc[-10:].corr().values[np.triu_indices(len(corr), k=1)]))
print('Cross-sectional std of 20d rets: %.4f' % R.iloc[-20:].sum().std())

# macro observation signals
print('\n--- Macro observation signals (as of 2026-07-15) ---')
for s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    df = pd.read_csv('../persistent/index_data/' + s + '.csv')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
    c = df['close']
    r20 = c.iloc[-1]/c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61] - 1 if len(c) > 61 else np.nan
    print('%-8s last=%.3f r20=%.2f%% r60=%.2f%%' % (s, c.iloc[-1], r20*100, r60*100))
