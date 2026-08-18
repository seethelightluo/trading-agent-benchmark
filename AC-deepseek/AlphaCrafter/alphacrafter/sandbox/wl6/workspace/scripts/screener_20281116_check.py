"""Screener cycle check: factor exposures & correlation as of 2028-11-15."""
import pandas as pd, numpy as np, os

base = '../persistent/stock_data'
assets = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
closes = {}
for a in assets:
    df = pd.read_csv(os.path.join(base, a + '.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[df['date'] <= '2028-11-15']
    closes[a] = df.set_index('date')['close']

panel = pd.DataFrame(closes).dropna()
print("panel rows:", len(panel), "cols:", panel.shape[1])
last = panel.iloc[-1]

def mom120(a):
    return last[a] / panel[a].iloc[-126] - 1

def beta_vix_neg(a):
    r_a = panel[a].pct_change().tail(60)
    vix = pd.read_csv('../persistent/index_data/VIX.csv')
    vix['date'] = pd.to_datetime(vix['date'])
    vix = vix[vix['date'] <= '2028-11-15'].set_index('date')['close']
    vix_r = vix.pct_change().tail(60)
    j = pd.concat([r_a, vix_r], axis=1, join='inner').dropna()
    if len(j) < 30:
        return np.nan
    cov = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    var = np.var(j.iloc[:, 1])
    return -cov / var if var > 0 else np.nan

def vol_beta_spx(a):
    r_a = panel[a].pct_change().tail(60)
    r_spx = panel['SPX'].pct_change().tail(60)
    j = pd.concat([r_a, r_spx], axis=1, join='inner').dropna()
    if len(j) < 30:
        return np.nan
    cov = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    var = np.var(j.iloc[:, 1])
    return cov / var if var > 0 else np.nan

def sign_ewma(a):
    c = panel[a]
    ew = c.ewm(span=60, adjust=False).mean()
    return 1.0 if last[a] > ew.iloc[-1] else -1.0

def dvr(a):
    r = panel[a].pct_change()
    dv = r[r < 0].tail(20).std()
    tv = r.tail(120).std()
    return dv / tv if tv > 0 else np.nan

def lowvol(a):
    return -panel[a].pct_change().tail(20).std()

fac = {}
fac['mom_120d_skip5'] = {a: mom120(a) for a in assets}
fac['beta_vix_60d_neg'] = {a: beta_vix_neg(a) for a in assets}
fac['vol_beta_spx_60d'] = {a: vol_beta_spx(a) for a in assets}
fac['sign_ewma_60d'] = {a: sign_ewma(a) for a in assets}
fac['down_vol_ratio_20x120'] = {a: dvr(a) for a in assets}
fac['low_vol_20d'] = {a: lowvol(a) for a in assets}

F = pd.DataFrame(fac)
print("\nCross-sectional exposure matrix (last date 2028-11-15):")
print(F.round(3).T)
print("\nPearson corr among selected factors:")
print(F.corr().round(2))

# Top-5 names by composite score (using ensemble weights)
ens = {
    'mom_120d_skip5': (0.28, 1),
    'beta_vix_60d_neg': (0.24, 1),
    'vol_beta_spx_60d': (0.18, 1),
    'sign_ewma_60d': (0.16, 1),
    'down_vol_ratio_20x120': (0.08, 1),
    'low_vol_20d': (0.06, -1),
}
score = {a: 0.0 for a in assets}
for fid, (w, d) in ens.items():
    s = F[fid].rank()  # 1..15
    for a in assets:
        score[a] += (w * d) * (s[a] - 0.5)
sc = pd.Series(score).sort_values(ascending=False)
print("\nComposite score ranking (top 10):")
print(sc.round(3).head(10))
