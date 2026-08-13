"""Screener: compute recent live cross-sectional IC for candidate factors."""
import pandas as pd, numpy as np, os

os.chdir('../persistent/stock_data')
assets = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

def load(a):
    df = pd.read_csv(a+'.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

px = pd.DataFrame({a: load(a)['close'].astype(float) for a in assets})
px = px[px.index <= '2031-06-19']
ret = px.pct_change()
logret = np.log(px).diff()

vix = pd.read_csv('../index_data/VIX.csv')
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.set_index('date').sort_index()['close'].astype(float)
vix = vix[vix.index <= '2031-06-19']

# factor signals
sig = pd.DataFrame(index=px.index)
sig['mom_120d_skip5'] = px.shift(5)/px.shift(125) - 1.0
sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
# vix_beta_cond: -beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20)-1)
vixr = vix.pct_change()
betas = ret.rolling(60).cov(vixr).div(vixr.rolling(60).var(), axis=0) if False else None
# compute beta per asset manually
beta = pd.DataFrame(index=px.index, columns=assets)
for a in assets:
    cov = ret[a].rolling(60).cov(vixr)
    var = vixr.rolling(60).var()
    beta[a] = cov/var
vix_move = vix/vix.shift(20) - 1.0
sig['vix_beta_cond_60x20'] = -beta.mul(vix_move, axis=0)
sig['miner2_20260715_nclv_1d'] = -(px - px.rolling(1).min().min(axis=1)) / (px.rolling(1).max().max(axis=1) - px.rolling(1).min().min(axis=1))
# nclv_1d: -(close - min(low,1))/(max(high,1)-min(low,1)); with close only approx:
sig['miner2_20260715_nclv_1d'] = -(px - px.min(axis=1)) / (px.max(axis=1) - px.min(axis=1))
sig['miner2_20260715_rev_2d'] = -(logret.shift(0) + logret.shift(1))
sig['miner2_20260715_rev_1d'] = -logret
sig['miner2_20260715_nclv_5d'] = -(px - px.rolling(5).min().min(axis=1)) / (px.rolling(5).max().max(axis=1) - px.rolling(5).min().min(axis=1))

# forward returns
fwd = {}
for h in [1,5,10]:
    fwd[h] = px.shift(-h)/px - 1.0

def rank_ic(s, fr, min_valid=8):
    out = []
    for dt in s.index:
        x = s.loc[dt]; y = fr.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < min_valid: continue
        out.append((dt, x[m].rank().corr(y[m].rank())))
    if not out: return pd.Series(dtype=float)
    return pd.Series(dict(out))

factors = ['mom_120d_skip5','vol_of_vol20x60','vix_beta_cond_60x20','miner2_20260715_nclv_1d','miner2_20260715_rev_2d','miner2_20260715_rev_1d','miner2_20260715_nclv_5d']

for f in factors:
    print('='*15, f)
    for h in [1,10]:
        ic = rank_ic(sig[f], fwd[h])
        for win, label in [(20,'20d'),(60,'60d'),(120,'120d')]:
            sub = ic.tail(win)
            if len(sub) < 5:
                print(f'  h{h} {label}: n={len(sub)} too few')
                continue
            m = sub.mean(); sd = sub.std(); icir = m/sd if sd>0 else np.nan
            print(f'  h{h} {label}: IC={m:+.4f} ICIR={icir:+.3f} hit={(sub>0).mean():.2f} n={len(sub)}')
    # signal flatness
    print('  last signal std:', sig[f].iloc[-1].std(), '| nonzero last:', (sig[f].iloc[-1].abs()>1e-12).sum())
