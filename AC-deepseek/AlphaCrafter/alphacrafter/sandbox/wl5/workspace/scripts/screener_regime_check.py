import pandas as pd, glob, os, numpy as np

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
rows = []
closes = {}
for f in files:
    name = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    closes[name] = df.set_index('date')['close']
    d = df[df['date'] <= pd.Timestamp('2027-08-25')].sort_values('date')
    c = d['close']
    r5 = c.iloc[-1] / c.iloc[-6] - 1 if len(c) > 6 else np.nan
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    vol20 = c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    hi60 = c.iloc[-60:].max()
    dd = c.iloc[-1] / hi60 - 1
    rows.append((name, d['date'].iloc[-1], r5, r20, r60, c.iloc[-1] / ma20 - 1, c.iloc[-1] / ma60 - 1, vol20, dd))

hdr = '{:<10s} {:>12s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>7s} {:>8s}'.format(
    'asset', 'last_date', 'r5', 'r20', 'r60', 'vsMA20', 'vsMA60', 'vol20', 'dd60')
print(hdr)
for r in rows:
    name, ld, r5, r20, r60, vm20, vm60, vol, dd = r
    print('{:<10s} {:>12s} {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>7.2f}% {:>6.1f}% {:>7.2f}%'.format(
        name, str(ld), r5 * 100, r20 * 100, r60 * 100, vm20 * 100, vm60 * 100, vol * 100, dd * 100))

# macro observation signals
print('\n--- MACRO OBSERVATION ---')
for f in sorted(glob.glob('../persistent/index_data/*.csv')):
    name = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date'])
    d = df[df['date'] <= pd.Timestamp('2027-08-25')].sort_values('date')
    c = d['close']
    r20 = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = c.iloc[-1] / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    vol20 = c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    print('{:<8s} last={} r20={:>7.2f}% r60={:>7.2f}% vol20={:>6.1f}%'.format(
        name, d['date'].iloc[-1], r20 * 100, r60 * 100, vol * 100))

# correlation regime: mean pairwise corr of 20d returns across the 15 assets (last 60d)
ret = pd.DataFrame({k: v.pct_change() for k, v in closes.items()}).dropna()
corr60 = ret.iloc[-60:].corr()
vals = corr60.values[np.triu_indices_from(corr60.values, k=1)]
print('\nmean pairwise corr (60d): {:.3f}'.format(np.nanmean(vals)))
corr120 = ret.iloc[-120:].corr()
vals120 = corr120.values[np.triu_indices_from(corr120.values, k=1)]
print('mean pairwise corr (120d): {:.3f}'.format(np.nanmean(vals120)))
