"""Explore cross-sectionally demeaned (relative) momentum factors at 10d horizon.
Universe: 15 tradable cross-asset instruments. Data restricted to sim visible_through (2031-02-05)."""
import pandas as pd, numpy as np

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VISIBLE = pd.Timestamp('2031-02-05')

data = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv', parse_dates=['date'])
    df = df[df['date'] <= VISIBLE].copy()
    df = df.sort_values('date').reset_index(drop=True)
    df['asset'] = a
    data[a] = df

close = pd.DataFrame({a: data[a].set_index('date')['close'] for a in ASSETS}).sort_index()

def ic_series(panel_vals, fwd, horizon, min_names=8):
    ics = []
    dates = sorted(panel_vals.index.intersection(fwd.index))
    for dt in dates:
        x = panel_vals.loc[dt].dropna()
        y = fwd.loc[dt].dropna()
        common = x.index.intersection(y.index)
        if len(common) < min_names:
            continue
        xv = x[common].astype(float)
        yv = y[common].astype(float)
        if xv.nunique() < 2 or yv.nunique() < 2:
            continue
        ics.append(pd.Series(yv).corr(pd.Series(xv), method='spearman'))
    ics = np.array(ics)
    ic = float(np.nanmean(ics))
    icir = float(np.nanmean(ics)/np.nanstd(ics)) if len(ics) > 1 else 0.0
    return ic, icir, len(ics)

# sanity: IB/benchmark admission gate uses absolute daily paper IC >= 0.0070 and ICIR >= 0.0840
fwd10 = close.pct_change(10).shift(-10)  # forward 10d simple returns

mom5 = close / close.shift(5) - 1
rel_mom5 = mom5.sub(mom5.median(axis=1), axis=0)
rel_mom5_mean = mom5.sub(mom5.mean(axis=1), axis=0)
mom10 = close / close.shift(10) - 1
rel_mom10 = mom10.sub(mom10.median(axis=1), axis=0)
mom20 = close / close.shift(20) - 1
rel_mom20 = mom20.sub(mom20.median(axis=1), axis=0)

for name, fac in [('abs_mom5', mom5),('rel_mom5_med', rel_mom5),('rel_mom5_mean', rel_mom5_mean),
                  ('rel_mom10_med', rel_mom10), ('rel_mom20_med', rel_mom20)]:
    ic, icir, n = ic_series(fac, fwd10, 10)
    print(f"{name}: IC={ic:.4f} ICIR={icir:.4f} n_dates={n}")

# Also check a volatility (realized vol 20d) factor and cross-tax demeaned vol
ret = close.pct_change()
vol20 = ret.rolling(20).std()
rel_vol20 = vol20.sub(vol20.median(axis=1), axis=0)
ic, icir, n = ic_series(rel_vol20, fwd10, 10)
print(f"rel_vol20_med: IC={ic:.4f} ICIR={icir:.4f} n_dates={n}")
ic, icir, n = ic_series(vol20, fwd10, 10)
print(f"abs_vol20: IC={ic:.4f} ICIR={icir:.4f} n_dates={n}")