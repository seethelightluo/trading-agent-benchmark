"""Screener: compute recent rank IC / ICIR / quality q for all active factors as of 2029-03-15.
Uses only data <= 2029-03-15 (current date 2029-03-16, last completed trading day 2029-03-15)."""
import pandas as pd, numpy as np, json, glob, os

CUTOFF = '2029-03-15'
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

def load_close():
    closes, opens, highs, lows = {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
        closes[a] = df['close']; opens[a] = df['open']; highs[a] = df['high']; lows[a] = df['low']
    C = pd.DataFrame(closes); O = pd.DataFrame(opens); H = pd.DataFrame(highs); L = pd.DataFrame(lows)
    return C, O, H, L

def load_vix():
    df = pd.read_csv('../persistent/index_data/VIX.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].set_index('date').sort_index()
    return df['close']

C, O, H, L = load_close()
VIX = load_vix()
ret = C.pct_change()
logret = np.log(C).diff()

def rolling_beta(y, x, win):
    # beta of y on x over trailing win (aligned to last row)
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    yv, xv = y.values, x.values
    res = np.full(yv.shape, np.nan)
    for i in range(win, len(yv)):
        xw = xv[i-win:i]
        xm = xw.mean(); ym = yv[i-win:i].mean(axis=0)
        num = ((yv[i-win:i]-ym) * (xw[:,None]-xm)).sum(axis=0)
        den = ((xw-xm)**2).sum()
        if den > 1e-12:
            res[i] = num/den
    out.iloc[:, :] = res
    return out

def rank_ic(sig, fwd):
    """sig: DataFrame (dates x assets) factor signals; fwd: DataFrame of forward returns aligned to same dates.
    Returns per-date spearman IC series."""
    ics = []
    idx = sig.index.intersection(fwd.index)
    for d in idx:
        s = sig.loc[d].dropna()
        f = fwd.loc[d].reindex(s.index).dropna()
        common = s.index.intersection(f.index)
        if len(common) < 5:
            continue
        sv = s[common].values; fv = f[common].values
        if np.nanstd(sv) == 0 or np.nanstd(fv) == 0:
            continue
        sr = pd.Series(sv).rank().values
        fr = pd.Series(fv).rank().values
        sr = sr - sr.mean(); fr = fr - fr.mean()
        denom = np.sqrt((sr**2).sum() * (fr**2).sum())
        if denom > 0:
            ics.append((d, (sr*fr).sum()/denom))
    if not ics:
        return pd.Series(dtype=float)
    return pd.Series(dict(ics))

def eval_factor(name, sig):
    res = {'factor_id': name}
    for h in (1, 5, 10):
        fwd = C.shift(-h) / C - 1.0  # forward h-day return
        ic_all = rank_ic(sig, fwd)
        if len(ic_all) == 0:
            res[f'ic{h}'] = None; res[f'icir{h}'] = None; res[f'n{h}'] = 0
            continue
        # recent windows
        for wl, tag in ((20,'20'), (60,'60'), (120,'120')):
            icw = ic_all.tail(wl)
            icm = icw.mean()
            ics = icw.std(ddof=1)
            res[f'ic{h}_{tag}'] = round(float(icm), 4)
            res[f'icir{h}_{tag}'] = round(float(icm/ics), 3) if ics and ics > 0 else None
        res[f'n{h}'] = int(len(ic_all))
    # full-window summary (last 250d for stability)
    ic1 = rank_ic(sig, C.shift(-1)/C - 1.0).tail(250)
    res['ic1_250'] = round(float(ic1.mean()), 4)
    res['icir1_250'] = round(float(ic1.mean()/ic1.std(ddof=1)), 3) if len(ic1) > 2 and ic1.std(ddof=1) > 0 else None
    return res

# ---- build signals ----
signals = {}
signals['mom_120d_skip5'] = C.shift(5) / C.shift(125) - 1.0
rv20 = ret.rolling(20).std()
signals['vol_of_vol20x60'] = rv20.rolling(60).std()
vixr = VIX.pct_change()
vix20 = VIX / VIX.shift(20) - 1.0
beta_vix = rolling_beta(logret, vixr, 60)
signals['vix_beta_cond_60x20'] = -beta_vix * vix20.to_numpy()[:, None]
signals['miner2_20260715_rev_1d'] = -(logret)
signals['miner2_20260715_rev_1d_vs'] = -(logret) / rv20
signals['miner2_20260715_rev_2d'] = -(np.log(C) - np.log(C).shift(2))
signals['miner2_20260715_rev_3d'] = -(np.log(C) - np.log(C).shift(3))
signals['miner2_20260715_rev_5d'] = -(np.log(C) - np.log(C).shift(5))
for w in (1, 2, 3, 5):
    rng = H.rolling(w).max() - L.rolling(w).min()
    signals[f'miner2_20260715_nclv_{w}d'] = -(C - L.rolling(w).min()) / rng
signals['miner2_20260715_nbody_1d'] = -(C - O) / (H - L)
signals['miner2_20260715_id_rev_1d'] = -(C / O - 1.0)

results = {}
for name, sig in signals.items():
    results[name] = eval_factor(name, sig)

# ---- crowding: average pairwise cross-sectional rank correlation (last 60d) ----
names = list(signals.keys())
last60 = C.index[-60:]
corr_mat = {}
for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        rhos = []
        for d in last60:
            s1 = signals[n1].loc[d].dropna(); s2 = signals[n2].loc[d].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 5: continue
            a = pd.Series(s1[common]).rank().values; b = pd.Series(s2[common]).rank().values
            a = a - a.mean(); b = b - b.mean()
            den = np.sqrt((a**2).sum()*(b**2).sum())
            if den > 0: rhos.append((a*b).sum()/den)
        if rhos:
            corr_mat[f'{n1}|{n2}'] = round(float(np.mean(rhos)), 3)

out = {'asof': CUTOFF, 'n_assets': len(ASSETS), 'factor_ics': results, 'pairwise_rank_corr_60d': corr_mat}
with open('screener_recent_ic.json', 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:6000])
