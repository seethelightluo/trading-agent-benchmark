import pandas as pd, numpy as np, os

base = '../persistent/stock_data'
assets = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
data = {}
for a in assets:
    df = pd.read_csv(os.path.join(base, a + '.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= '2030-09-20'].set_index('date')['close']
    data[a] = df

ret = pd.DataFrame({a: data[a].pct_change() for a in assets})

# 1) rate_beta_cn10y_60d: beta of each asset's returns to CN10Y returns over last 60d
cn = ret['CN10Y'].iloc[-60:]
print('CN10Y 60d return std:', cn.std(), ' nonzero days:', int((cn.abs() > 1e-12).sum()))
beta = pd.Series({a: np.cov(ret[a].iloc[-60:].fillna(0), cn.fillna(0))[0, 1] / (cn.var() + 1e-12) for a in assets})
print('\nrate_beta_cn10y_60d approx cross-section (all ~0 if CN10Y frozen):')
print(beta.round(4).to_string())

# 2) vol_adj_mom_accel_20x60
def mom_accel(c):
    r = c.pct_change()
    m20 = c.iloc[-1] / c.iloc[-21] - 1
    m60 = c.iloc[-1] / c.iloc[-61] - 1
    v = r.iloc[-20:].std()
    return (m20 - m60) / v if v > 0 else np.nan

ma = pd.Series({a: mom_accel(data[a]) for a in assets})
print('\nvol_adj_mom_accel_20x60 approx cross-section:')
print(ma.round(4).to_string())

# 3) dn_mkt_beta_60d approx: beta of each asset to market downside days (equal-weight live names)
live = ['000688.SH','COPPER','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
mkt = ret[live].mean(axis=1)
down = mkt[mkt < 0]

def dnbeta(c):
    r = c.pct_change()
    common = pd.concat([r, down], axis=1, join='inner').dropna()
    if len(common) < 20:
        return np.nan
    rr = common.iloc[:, 0]; mm = common.iloc[:, 1]
    return rr.cov(mm) / (mm.var() + 1e-12)

db = pd.Series({a: dnbeta(data[a]) for a in assets})
print('\ndn_mkt_beta_60d approx (vs equal-weight live-market downside):')
print(db.round(4).to_string())

# cross-sectional dispersion & correlation (live assets, 60d)
lr = ret[live].iloc[-60:]
disp = lr.std(axis=1).mean() * np.sqrt(252)
corr = lr.corr()
avg_corr = (corr.values[np.triu_indices(len(live), 1)]).mean()
print(f'\nLive-asset 60d cross-sectional dispersion (ann.): {disp:.3f}')
print(f'Live-asset 60d avg pairwise correlation: {avg_corr:.3f}')
