"""miner_1 2026-07-16: broad factor screen for recovery cycle.

Screens a diverse battery of cross-asset factor families on the 15-instrument
tradable universe (2020-01-01 .. 2026-07-15 visible).  Computes daily paper
rank IC vs 1d forward return, ICIR, hit, coverage, turnover, decay and
by-year stability.  Reports pairwise mean cross-sectional |spearman| among
top candidates and vs the pending nclv_1d library artifact (2388x15 grid).
"""
import numpy as np
import pandas as pd
import json, os

panel = pd.read_pickle('scripts/panel_cache.pkl')
close, open_, high, low, vol, ret = (panel[k] for k in
                                     ['close', 'open', 'high', 'low', 'vol', 'ret'])
macro = panel['macro'].reindex(close.index, method='ffill')
idx = close.index
SYMBOLS = list(close.columns)
N = len(idx)

def fdf(series_dict):
    return pd.DataFrame(series_dict, index=idx, columns=SYMBOLS)

F = {}
# ---------- trend / momentum ----------
F['mom_10_skip5']   = close.shift(5) / close.shift(15) - 1.0
F['mom_60_skip5']   = close.shift(5) / close.shift(65) - 1.0
F['mom_120_skip5']  = close.shift(5) / close.shift(125) - 1.0
F['mom_20']         = close / close.shift(20) - 1.0
# residual (breadth-relative) momentum
F['resid_mom_60']   = F['mom_60_skip5'].sub(F['mom_60_skip5'].mean(axis=1), axis=0)
# breadth-confirmed momentum: 20d return * share of positive days (20d)
pos_share_20 = (ret > 0).rolling(20).mean()
F['breadth_mom_20'] = F['mom_20'] * pos_share_20
# efficiency ratio 60d (directional efficiency)
absret = ret.abs()
F['eff_ratio_60'] = (close - close.shift(60)).abs() / absret.rolling(60).sum()
# close location in daily range, 20d mean
rng = (high - low).replace(0, np.nan)
F['range_loc_20'] = ((close - low) / rng).rolling(20).mean()
# proximity to 20d high
F['hilo_pos_20'] = close / high.rolling(20).max() - 1.0
# ---------- volatility ----------
std5  = ret.rolling(5).std()
std20 = ret.rolling(20).std()
std60 = ret.rolling(60).std()
F['vol_ratio_5_60']  = std5 / std60
F['vol_ratio_20_60'] = std20 / std60
F['vol_of_vol_20_60'] = std20.rolling(60).std()
neg = ret.where(ret < 0, 0.0)
semi20 = np.sqrt((neg ** 2).rolling(20).mean())
F['downside_vol_ratio_20'] = semi20 / std20
# ---------- cross-asset basket betas ----------
eq = SYMBOLS[:8]
comm = ['XAU', 'COPPER', 'WTI']
crypto = ['BTC', 'ETH']
basket_eq = ret[eq].mean(axis=1)
basket_comm = ret[comm].mean(axis=1)
basket_crypto = ret[crypto].mean(axis=1)
basket_all = ret.mean(axis=1)

def roll_beta(y, x, w=60):
    out = {}
    for s in y.columns:
        df = pd.concat([y[s].rename('y'), x.rename('x')], axis=1)
        b = df['y'].rolling(w).cov(df['x']) / df['x'].rolling(w).var()
        out[s] = b
    return fdf(out)

F['beta_eq_60']   = roll_beta(ret, basket_eq, 60)
F['beta_comm_60'] = roll_beta(ret, basket_comm, 60)
F['beta_crypto_60'] = roll_beta(ret, basket_crypto, 60)
# downside participation: beta on days basket falls minus beta on up days
def cond_beta_diff(y, x, w=60):
    out = {}
    for s in y.columns:
        d = pd.concat([y[s].rename('y'), x.rename('x')], axis=1)
        down = d[d['x'] < 0]
        up = d[d['x'] > 0]
        bd = down['y'].rolling(w).cov(down['x']) / down['x'].rolling(w).var()
        bu = up['y'].rolling(w).cov(up['x']) / up['x'].rolling(w).var()
        out[s] = bd - bu
    return fdf(out)
F['downside_part_60'] = cond_beta_diff(ret, basket_all, 60)
# ---------- macro betas ----------
m_ret = macro.pct_change()
for mname in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    F[f'beta_{mname}_60'] = roll_beta(ret, m_ret[mname], 60)
    up = m_ret[mname].where(m_ret[mname] > 0)
    d = pd.concat([m_ret[mname].rename('m'), up.rename('up')], axis=1)
    F[f'beta_{mname}_up_60'] = roll_beta(ret, d['up'], 60)  # NaN where not up
# yield spread beta (US10Y - CN10Y change)
us10y = close['US10Y']; cn10y = close['CN10Y']
spread = us10y - cn10y
spread_chg = spread.diff()
F['beta_yspread_60'] = roll_beta(ret, spread_chg, 60)
# ---------- cross-asset correlation concentration ----------
def corr_conc(w=60):
    out = {}
    for s in SYMBOLS:
        r = ret[s]
        others = [c for c in SYMBOLS if c != s]
        cc = {}
        for o in others:
            cc[o] = r.rolling(w).corr(ret[o])
        out[s] = pd.concat(cc.values(), axis=1).abs().mean(axis=1)
    return fdf(out)
F['corr_conc_60'] = corr_conc(60)
# ---------- volume / liquidity ----------
F['volz_20'] = vol / vol.rolling(20).mean() - 1.0
F['vol_trend_10_60'] = vol.rolling(10).mean() / vol.rolling(60).mean() - 1.0
# ---------- jump imbalance ----------
thr = 2.0 * std20
big_up = ((ret > thr) * 1.0).rolling(20).sum()
big_dn = ((ret < -thr) * 1.0).rolling(20).sum()
F['jump_imb_20'] = big_up - big_dn
# tail severity: mean of worst-5% daily returns over 60d (signed)
def tail_sev(w=60, q=0.05):
    out = {}
    for s in SYMBOLS:
        r = ret[s].rolling(w).apply(
            lambda x: x[x < np.nanpercentile(x, q * 100)].mean() if len(x) >= w else np.nan,
            raw=True)
        out[s] = r
    return fdf(out)
F['tail_sev_60'] = tail_sev(60)
# overnight/intraday dislocation
overnight = open_ / close.shift(1) - 1.0
intraday = close / open_ - 1.0
F['gap_rev_20'] = (overnight * intraday).rolling(20).mean()

# ---------- validation ----------
fwd1 = close.shift(-1) / close - 1.0
def daily_rank_ic(fact, fwd, min_names=8):
    ics, dates, obs = [], [], []
    for dt in fact.index:
        f = fact.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_names:
            continue
        x = f[common].astype(float).rank()
        y = r[common].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic); dates.append(dt); obs.append(len(common))
    return np.array(ics), np.array(dates), np.array(obs)

def turnover_10d(fact):
    f = fact.dropna(how='all')
    if len(f) < 40:
        return np.nan
    idxs = f.index[::10]
    prev = None; tot, cnt = 0.0, 0
    for dt in idxs:
        row = f.loc[dt].dropna()
        if len(row) < 5:
            continue
        r = row.rank()
        if prev is not None:
            common = r.index.intersection(prev.index)
            if len(common) >= 5:
                tot += float((r[common] - prev[common]).abs().mean()) / (len(common) - 1)
                cnt += 1
        prev = r
    return tot / cnt if cnt else np.nan

def mean_abs_spearman(A, B):
    vals = []
    for i in range(min(len(A), len(B))):
        a, b = A[i], B[i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 8:
            continue
        ra = pd.Series(a[m]).rank().values
        rb = pd.Series(b[m]).rank().values
        if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(ra, rb)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan

results = []
for name, fact in F.items():
    ics, dates, obs = daily_rank_ic(fact, fwd1)
    if len(ics) < 100:
        print(f"[skip] {name}: only {len(ics)} IC dates"); continue
    ic = ics.mean(); icir = ic / ics.std(ddof=1) if ics.std(ddof=1) > 0 else np.nan
    hit = float((ics > 0).mean())
    cov = float(fact.notna().mean().mean())
    to = turnover_10d(fact)
    # decay
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwd = close.shift(-h) / close - 1.0
        ih, _, _ = daily_rank_ic(fact, fwd)
        decay[h] = float(ih.mean()) if len(ih) else np.nan
    # by-year
    by_year = {}
    yrs = pd.Series(dates).dt.year.unique()
    for y in yrs:
        m = pd.Series(dates).dt.year == y
        by_year[str(int(y))] = round(float(ics[m].mean()), 4)
    results.append(dict(name=name, ic=ic, icir=icir, quality=abs(ic)*abs(icir),
                        hit=hit, cov=cov, to=to, n_dates=len(ics),
                        decay=decay, by_year=by_year))

res = pd.DataFrame(results).sort_values('quality', ascending=False)
pd.set_option('display.width', 200)
print("\n=== SCREEN RESULTS (sorted by |IC*ICIR|) ===")
print(res[['name', 'ic', 'icir', 'quality', 'hit', 'cov', 'to', 'n_dates']].to_string(index=False))
print("\n=== by-year IC for top 15 ===")
for _, r in res.head(15).iterrows():
    print(r['name'], r['by_year'])
print("\n=== decay for top 15 ===")
for _, r in res.head(15).iterrows():
    print(r['name'], {k: round(v, 4) for k, v in r['decay'].items()})
res.to_json('scripts/_screen_recovery_results.json', orient='records')

# ---------- pairwise rho among top candidates ----------
top = res.head(12)['name'].tolist()
print("\n=== pairwise mean abs spearman (top 12) ===")
mat = {}
for a in top:
    mat[a] = {}
    for b in top:
        if b == a:
            mat[a][b] = 1.0
        else:
            mat[a][b] = round(mean_abs_spearman(F[a].values, F[b].values), 3)
print("names:", top)
for a in top:
    print(a, [mat[a][b] for b in top])

# vs nclv_1d pending artifact
nclv = np.load('factors/miner2_20260716_nclv_1d.npy', allow_pickle=False)
print("\n=== rho vs nclv_1d (2388x15 grid) ===")
for a in top:
    rho = mean_abs_spearman(F[a].values.astype(float), nclv.astype(float))
    print(f"{a}: {rho:.3f}")
