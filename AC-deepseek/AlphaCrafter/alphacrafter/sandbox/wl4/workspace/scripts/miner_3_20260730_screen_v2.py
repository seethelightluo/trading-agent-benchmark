"""miner_3 candidate factor screening v2 - robust pipeline.

Fixes the weekend-NaN contamination: rolling/cov windows are computed per-asset on
each asset's own clean series (dropna), then reindexed back to the union calendar.
Cross-sectional rank IC vs 10d forward returns, 2020-01-01..2026-07-15, min 8 valid.
Admission gates (15-asset universe): |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WL = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
END = pd.Timestamp('2026-07-15')
HORIZON = 10
MIN_ASSETS = 8


def load_asset(sym):
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


frames = {s: load_asset(s) for s in WL}
# union calendar
idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values() if f is not None])))
closes = pd.DataFrame({s: frames[s]['close'].reindex(idx) for s in WL})
rets = closes.pct_change()
volume = pd.DataFrame({s: frames[s]['volume'].reindex(idx) for s in WL})
opens = pd.DataFrame({s: frames[s]['open'].reindex(idx) for s in WL})

# macro signals
mac = {}
for m in ['VIX', 'DXY', 'USDJPY', 'EURUSD', 'USDCNY']:
    p = f"../persistent/index_data/{m}.csv"
    d = pd.read_csv(p)
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['date'] <= END].set_index('date')
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    mac[m] = d['close'].reindex(idx)
    mac[m + '_ret'] = mac[m].pct_change()

vix = mac['VIX']; dxy = mac['DXY']; jpy = mac['USDJPY']; eur = mac['EURUSD']
print(f"universe {len(WL)} assets, calendar n={len(idx)}, "
      f"weekday dates with >=8 assets: {int(((closes.notna().sum(axis=1) >= 8) & (idx.dayofweek < 5)).sum())}")

fwd = closes.shift(-HORIZON) / closes - 1.0

# ---------------- library signals (for corr check) ----------------
def clean_series(x, win, func, min_pct=0.5):
    """Per-asset clean rolling to avoid weekend-NaN contamination."""
    out = {}
    for a in x.columns:
        s = x[a].dropna()
        if len(s) < win * min_pct:
            out[a] = pd.Series(np.nan, index=x.index)
            continue
        r = func(s, win)
        out[a] = r.reindex(x.index)
    return pd.DataFrame(out, index=x.index)


def rstd(s, w): return s.rolling(w).std()
def rmean(s, w): return s.rolling(w).mean()

LIB = {}
LIB['mom_10d_skip5'] = closes.shift(5) / closes.shift(15) - 1.0
LIB['mom_120d_skip5'] = closes.shift(5) / closes.shift(125) - 1.0
LIB['vol_of_vol20x60'] = clean_series(rets, 20, lambda s, w: s.rolling(w).std().rolling(60).std())
# vix beta conditional
b = pd.DataFrame(index=idx, columns=WL, dtype=float)
for a in WL:
    d = pd.concat([rets[a].rename('a'), mac['VIX_ret'].rename('v')], axis=1).dropna()
    if len(d) < 120:
        continue
    beta = (d['a'].rolling(60).cov(d['v']) / d['v'].rolling(60).var().replace(0, np.nan)).reindex(idx)
    b[a] = -beta * (vix / vix.shift(20) - 1.0)
LIB['vix_beta_cond_60x20'] = b

# ---------------- IC machinery ----------------
def ic_series(fp):
    out = {}
    for t in fp.index:
        if t.dayofweek >= 5:
            continue
        fv = fp.loc[t]
        fr = fwd.loc[t].reindex(fv.index)
        m = fv.notna() & fr.notna()
        if int(m.sum()) < MIN_ASSETS:
            continue
        ic, _ = spearmanr(fv[m], fr[m])
        if np.isfinite(ic):
            out[t] = ic
    return pd.Series(out)


def max_lib_corr(fp):
    best = (0.0, None)
    for lid, lp in LIB.items():
        both = pd.concat([fp.stack().rename('c'), lp.stack().rename('l')], axis=1).dropna()
        if len(both) < 500:
            continue
        r = float(both['c'].corr(both['l']))
        if abs(r) > best[0]:
            best = (abs(r), lid)
    return best


def analyze(name, fp):
    ic = ic_series(fp)
    if len(ic) < 200:
        return dict(name=name, n_ic=len(ic), note='insufficient dates')
    icir = ic.mean() / ic.std(ddof=1)
    hit = float((np.sign(ic) == np.sign(ic.mean())).mean())
    valid = fp.notna()
    cov_ad = float(valid.sum().sum() / (len(fp) * len(fp.columns)))
    cov_d8 = float(((valid.sum(axis=1) >= MIN_ASSETS) & (fp.index.dayofweek < 5)).mean())
    ranks = fp.rank(axis=1)
    to = float((ranks - ranks.shift(10)).abs().mean().mean())
    rho, rkey = max_lib_corr(fp)
    return dict(name=name, ic=round(float(ic.mean()), 4), icir=round(float(icir), 4),
                hit=round(hit, 3), n_ic=len(ic), cov_ad=round(cov_ad, 3), cov_d8=round(cov_d8, 3),
                to=round(to, 3), rho=round(rho, 3), rkey=rkey)


# ---------------- candidates ----------------
C = {}
C['eff_ratio_20d'] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
C['eff_ratio_60d'] = (closes - closes.shift(60)).abs() / rets.abs().rolling(60).sum()
C['range_pos_10d'] = (closes - closes.rolling(10).min()) / (closes.rolling(10).max() - closes.rolling(10).min())
C['range_pos_20d'] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
C['w52high_prox'] = closes / closes.rolling(252).max() - 1.0
C['dd_prox_60d'] = closes / closes.rolling(60).max() - 1.0
C['vol_adj_mom_60d'] = (closes / closes.shift(60) - 1.0) / clean_series(rets, 20, rstd)
C['vol_adj_mom_20d'] = (closes / closes.shift(20) - 1.0) / clean_series(rets, 20, rstd)
C['amihud_20d'] = (rets.abs() / (volume + 1.0)).rolling(20).mean()
C['vol_trend_20_60'] = clean_series(volume, 20, rmean) / clean_series(volume, 60, rmean)
C['downside_sd_60d'] = clean_series(rets.clip(upper=0), 60, lambda s, w: np.sqrt((s ** 2).rolling(w).mean()))
C['skew_20d'] = clean_series(rets, 20, lambda s, w: s.rolling(w).skew())
C['hl_range_20d'] = (closes.rolling(20).max() - closes.rolling(20).min()) / closes
C['maxdd_60d'] = (closes / closes.rolling(60).max() - 1.0).rolling(60).min()
C['autocorr_5d'] = clean_series(rets, 60, lambda s, w: s.rolling(w).apply(lambda z: pd.Series(z).autocorr(1) if len(z) > 5 else np.nan, raw=False))
C['vol_ratio_10_60'] = clean_series(rets, 10, rstd) / clean_series(rets, 60, rstd)
C['updown_ratio_20d'] = rets.clip(lower=0).rolling(20).sum().abs() / (rets.clip(upper=0).rolling(20).sum().abs() + 1e-9)
C['rsi_14'] = 100 - 100 / (1 + rets.clip(lower=0).rolling(14).mean() / (rets.clip(upper=0).abs().rolling(14).mean() + 1e-9))
C['vol_zscore_1y'] = (clean_series(rets, 20, rstd) - clean_series(rets, 20, rstd).rolling(252).mean()) / clean_series(rets, 20, rstd).rolling(252).std()
C['sma_slope_60d'] = (closes.rolling(60).mean() / closes.rolling(60).mean().shift(20) - 1.0)
C['overnight_gap_20d'] = (opens / closes.shift(1) - 1.0).rolling(20).mean()
C['gap_zscore_10d'] = ((opens / closes.shift(1) - 1.0).rolling(10).mean() / (opens / closes.shift(1) - 1.0).rolling(10).std())

# yield-spread factors (levels are yields)
us10 = closes['US10Y']; cn10 = closes['CN10Y']
spread = us10 - cn10
C['yld_spread_level'] = spread.to_frame('US10Y')  # constant per date across assets -> no cross-section
del C['yld_spread_level']

# macro beta-conditional factors
def cond_beta(cond, cond_ret, sign):
    b = pd.DataFrame(index=idx, columns=WL, dtype=float)
    for a in WL:
        d = pd.concat([rets[a].rename('a'), cond_ret.rename('c')], axis=1).dropna()
        if len(d) < 120:
            continue
        beta = (d['a'].rolling(60).cov(d['c']) / d['c'].rolling(60).var().replace(0, np.nan)).reindex(idx)
        b[a] = sign * beta * (cond / cond.shift(20) - 1.0)
    return b

C['dxy_beta_cond_60x20'] = cond_beta(dxy, mac['DXY_ret'], -1)
C['usdjpy_beta_cond_60x20'] = cond_beta(jpy, mac['USDJPY_ret'], 1)
C['eurusd_beta_cond_60x20'] = cond_beta(eur, mac['EURUSD_ret'], -1)
C['vix_hedge_cond_60x20'] = cond_beta(vix, mac['VIX_ret'], 1)  # +beta*vixmove = risk-on tilt

# cross-asset correlation: rolling 60d mean pairwise corr of each asset vs others
def cross_corr(win=60):
    out = pd.DataFrame(index=idx, columns=WL, dtype=float)
    r = rets[WL]
    for t in idx:
        w = r.loc[:t].tail(win)
        if len(w) < 30:
            continue
        c = w.corr()
        for a in WL:
            others = c[a].drop(index=a)
            out.loc[t, a] = others.mean()
    return out

C['cross_corr_60d'] = cross_corr(60)

results = []
for name, fp in C.items():
    res = analyze(name, fp)
    results.append(res)
    if 'note' in res:
        print(f"{name:26s} NOTE={res['note']}")
        continue
    ok = abs(res['ic']) >= 0.007 and abs(res['icir']) >= 0.084
    flag = 'PASS' if ok else ''
    print(f"{name:26s} IC={res['ic']:>8} ICIR={res['icir']:>8} hit={res['hit']} n={res['n_ic']} "
          f"covAD={res['cov_ad']} covD8={res['cov_d8']} to={res['to']} rho={res['rho']}({res['rkey']}) {flag}")

print("\n--- sorted by |ICIR| ---")
for r in sorted(results, key=lambda x: -abs(x.get('icir', 0))):
    print(r)
