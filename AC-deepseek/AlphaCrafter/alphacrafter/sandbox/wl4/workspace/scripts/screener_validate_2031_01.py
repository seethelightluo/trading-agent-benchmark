"""SCREENER validation script (2031-01-27 cycle). v2: truncate at visible_through 2031-01-24."""
import json
import numpy as np
import pandas as pd

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
DATA_DIR = '../persistent/stock_data/'
CUTOFF = '2031-01-24'

prices = {}
for a in ASSETS:
    df = pd.read_csv(f'{DATA_DIR}/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    prices[a] = df['close']
px = pd.DataFrame(prices).sort_index()
px = px[~px.index.duplicated(keep='last')]
px = px[px.index <= CUTOFF]
print('price panel:', px.shape, px.index.min().date(), '->', px.index.max().date())

ret = px.pct_change()
mkt_ret = ret.mean(axis=1)

# ---------- factor 1: vol_adj_mom_accel_20x60 ----------
mom20 = px / px.shift(20) - 1.0
mom60 = px / px.shift(60) - 1.0
vol20 = ret.rolling(20).std()
sig_mom = (mom20 - mom60) / vol20

# ---------- factor 2 & 3: rolling beta ----------
def rolling_beta(asset_ret, mkt, win=60, min_obs=40):
    out = pd.DataFrame(np.nan, index=asset_ret.index, columns=asset_ret.columns)
    for i in range(win, len(asset_ret)):
        x = mkt.iloc[i-win:i]
        for c in asset_ret.columns:
            y = asset_ret[c].iloc[i-win:i]
            mask = x.notna() & y.notna()
            if mask.sum() < min_obs:
                continue
            xv = x[mask].values; yv = y[mask].values
            if xv.std() < 1e-12:
                continue
            out[c].iloc[i] = np.cov(xv, yv)[0, 1] / xv.var()
    return out

down = mkt_ret.where(mkt_ret < 0)
sig_dn = rolling_beta(ret, down)

cn10y = px['CN10Y'].pct_change()
sig_rate = rolling_beta(ret, cn10y)

signals = {'vol_adj_mom_accel_20x60': sig_mom,
           'dn_mkt_beta_60d': sig_dn,
           'rate_beta_cn10y_60d': sig_rate}

# ---------- rank IC at h=10 ----------
fwd = px.shift(-10) / px - 1.0

def rank_ic(sig, fwd, n_max=None):
    ics = []
    for dt in sig.index:
        x = sig.loc[dt]; y = fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < 8:
            continue
        r = np.corrcoef(x[m].rank(), y[m].rank())[0, 1]
        if np.isfinite(r):
            ics.append((dt, r))
    if not ics:
        return None
    ic = pd.Series([r for _, r in ics], index=[d for d, _ in ics])
    if n_max is not None and len(ic) > n_max:
        ic = ic.iloc[-n_max:]
    return ic

print('\n===== Rank IC (h=10) =====')
for name, sig in signals.items():
    ic_full = rank_ic(sig, fwd)
    if ic_full is None:
        print(f'{name:28s} no valid IC dates'); continue
    print(f'{name:28s} FULL     n={len(ic_full):4d} IC={ic_full.mean():+.4f} ICIR={ic_full.mean()/ic_full.std():+.3f} hit={((ic_full>0).mean()):.3f}')
    for n in (750, 500, 250, 120):
        ic = rank_ic(sig, fwd, n_max=n)
        if ic is not None and len(ic) > 0:
            print(f'{"":28s} LAST{n:<5d} n={len(ic):4d} IC={ic.mean():+.4f} ICIR={ic.mean()/ic.std():+.3f} hit={((ic>0).mean()):.3f}')

# ---------- regime ----------
print('\n===== Regime (as of 2031-01-24) =====')
for a in ['SPX', '000300.SH', 'NDX', 'SOX', 'XAU', 'BTC', 'WTI', 'COPPER', 'CN10Y', 'US10Y', 'HSI']:
    c = px[a].dropna()
    if len(c) < 210:
        print(f'{a:10s} insufficient history ({len(c)})'); continue
    ma200 = c.rolling(200).mean(); ma60 = c.rolling(60).mean()
    cur = c.iloc[-1]; m200 = ma200.iloc[-1]; m60 = ma60.iloc[-1]
    r60 = c.iloc[-1] / c.iloc[-61] - 1
    r20 = c.iloc[-1] / c.iloc[-21] - 1
    print(f'{a:10s} last={cur:12.4f} vs200d={cur/m200-1:+7.2%} vs60d={cur/m60-1:+7.2%} ret60d={r60:+8.2%} ret20d={r20:+8.2%}')

try:
    vix = pd.read_csv('../persistent/index_data/VIX.csv')
    vix['date'] = pd.to_datetime(vix['date'])
    vix = vix.set_index('date').sort_index()
    vix = vix[vix.index <= CUTOFF]
    vcol = 'close' if 'close' in vix.columns else vix.columns[1]
    vixv = vix[vcol].dropna()
    print(f'VIX last={vixv.iloc[-1]:.2f} mean60d={vixv.iloc[-60:].mean():.2f} max60d={vixv.iloc[-60:].max():.2f} pct90={vixv.iloc[-250:].quantile(0.9):.2f}')
except Exception as e:
    print('VIX err', e)

r = ret.iloc[-60:]
cc = r.corr().abs(); np.fill_diagonal(cc.values, np.nan)
print(f'avg pairwise |corr| 60d = {cc.stack().mean():.3f}')
r2 = ret.iloc[-250:]
cc2 = r2.corr().abs(); np.fill_diagonal(cc2.values, np.nan)
print(f'avg pairwise |corr| 250d = {cc2.stack().mean():.3f}')

print(f'mkt realized vol 20d (ann) = {mkt_ret.iloc[-20:].std()*np.sqrt(252):.2%}')
print(f'mkt realized vol 60d (ann) = {mkt_ret.iloc[-60:].std()*np.sqrt(252):.2%}')

mkt = px.mean(axis=1)
mkt_ma200 = mkt.rolling(200).mean()
print(f'EW-mkt last/200dMA = {mkt.iloc[-1]/mkt_ma200.iloc[-1]-1:+.2%}')
print(f'EW-mkt ret60d = {mkt.iloc[-1]/mkt.iloc[-61]-1:+.2%} ret20d={mkt.iloc[-1]/mkt.iloc[-21]-1:+.2%}')

print('\nlast-10d asset returns (2031-01-13..01-24):')
print((px.iloc[-1] / px.iloc[-11] - 1).round(4).to_string())

print('\nlast-60d asset returns:')
print((px.iloc[-1] / px.iloc[-61] - 1).round(4).to_string())

# dispersion
print(f'avg cross-sectional dispersion 20d = {ret.iloc[-20:].std(axis=1).mean():.4f}')

# factor pairwise correlation
print('\n===== Factor signal pairwise spearman rho (last 250d, cross-sectional) =====')
import scipy.stats as st
for n1 in signals:
    for n2 in signals:
        if n1 >= n2:
            continue
        s1 = signals[n1].iloc[-250:]; s2 = signals[n2].iloc[-250:]
        vals = []
        for dt in s1.index:
            x = s1.loc[dt]; y = s2.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 8:
                vals.append(st.spearmanr(x[m], y[m]).statistic)
        print(f'{n1} vs {n2}: mean rho={np.mean(vals):+.3f} (n={len(vals)})')
