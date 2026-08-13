"""miner1 2031-09-26: screen candidate factor families across the 15-asset cross-section.

Uses panel cache built by miner1_20310926_build_panel.py. Computes per-date cross-sectional
Spearman IC at horizons 1/5/10 for a battery of interpretable candidates. Reports mean IC,
ICIR (mean/std), hit ratio, coverage. Excludes flat-artifact assets (HSI, CN10Y) from IC obs.
"""
import pandas as pd, numpy as np, pathlib, json

panel = pd.read_pickle('scripts/panel_cache_20310926.pkl')
close = panel['close']
vol = panel['vol']
ret = panel['ret']
macro = panel['macro']

# drop assets with no variance (flat artifacts) from IC computations
FLAT = ['HSI', 'CN10Y']
rets = ret.drop(columns=[c for c in FLAT if c in ret.columns])
px = close.drop(columns=[c for c in FLAT if c in close.columns])

def rolling_apply(df, fn, win, minp=None):
    return df.rolling(win, min_periods=minp if minp else win).apply(fn, raw=True)

# ---------- candidate factor definitions (returns DataFrame, aligned to close) ----------
cands = {}

# 1) Trend efficiency: R^2 of linear trend over 60d (clean-trend persistence)
def _trend_eff(x):
    n = len(x)
    if n < 5 or np.isnan(x).any():
        return np.nan
    t = np.arange(n)
    if np.nanstd(x) == 0 or np.std(x) == 0:
        return np.nan
    c = np.corrcoef(t, x)[0, 1]
    return c * c
cands['trend_eff_60'] = px.rolling(60).apply(_trend_eff, raw=True)

# 2) 20d return skewness (lottery/downside risk)
cands['skew_20'] = ret.drop(columns=[c for c in FLAT if c in ret.columns]).rolling(20).skew()

# 3) Range position: (close - min)/(max - min) over 20d
low20 = px.rolling(20).min()
high20 = px.rolling(20).max()
cands['range_pos_20'] = (px - low20) / (high20 - low20).replace(0, np.nan)

# 4) Drawdown from 252d high
cands['dd_252'] = px / px.rolling(252).max() - 1.0

# 5) Vol ratio 20d/60d realized vol
rv20 = ret.drop(columns=[c for c in FLAT if c in ret.columns]).rolling(20).std()
rv60 = ret.drop(columns=[c for c in FLAT if c in ret.columns]).rolling(60).std()
cands['vol_ratio_20_60'] = rv20 / rv60

# 6) Downside vol ratio: semi-deviation (negative returns) / total vol over 60d
def _downside_ratio(x):
    neg = x[x < 0]
    if len(neg) < 5 or np.std(x) == 0 or np.isnan(x).any():
        return np.nan
    return np.std(neg) / np.std(x)
cands['downside_vol_60'] = ret.drop(columns=[c for c in FLAT if c in ret.columns]).rolling(60).apply(_downside_ratio, raw=True)

# 7) Rate sensitivity: 60d correlation of asset returns with US10Y returns (only for assets w/ data)
if 'US10Y' in ret.columns:
    r_us10y = ret['US10Y']
    corr_rates = ret.apply(lambda s: s.rolling(60).corr(r_us10y))
    corr_rates = corr_rates.drop(columns=['US10Y'], errors='ignore')
    corr_rates = corr_rates.reindex(px.columns)
    cands['rate_beta_60'] = corr_rates

# 8) RSI-style oscillator (14d): strength of recent up-moves
def _rsi(x):
    d = np.diff(x)
    if len(d) < 10:
        return np.nan
    up = d[d > 0].mean() if (d > 0).any() else 0.0
    dn = -d[d < 0].mean() if (d < 0).any() else 0.0
    if up + dn == 0:
        return 50.0
    return 100.0 * up / (up + dn)
cands['rsi_14'] = px.rolling(15).apply(_rsi, raw=True)

# 9) Volume trend: 10d mean volume / 60d mean volume
cands['vol_trend_10_60'] = vol.rolling(10).mean() / vol.rolling(60).mean()

# 10) Max drawdown over 60d
def _max_dd(x):
    if len(x) < 20 or np.isnan(x).any():
        return np.nan
    peak = np.maximum.accumulate(x)
    return (x / peak - 1.0).min()
cands['maxdd_60'] = px.rolling(60).apply(_max_dd, raw=True)

# 11) 5d reversal of 20d momentum (short-term pullback in a trend)
cands['rev5x_mom20'] = -(px.pct_change(5) - px.pct_change(20))

# 12) Cross-asset breadth: rank of 20d return among the 15 assets (dispersion/lead-lag)
mom20 = px.pct_change(20)
cands['mom20_rank'] = mom20.rank(axis=1, pct=True)

# ---------- IC evaluation ----------
def forward_returns(px, h):
    return px.shift(-h) / px - 1.0

def eval_factor(name, f, px, h=10, min_obs=8):
    fwd = forward_returns(px, h)
    f_al = f.reindex(px.index)
    fwd_al = fwd.reindex(px.index)
    ics = []
    valid_dates = 0
    for dt in px.index:
        fv = f_al.loc[dt]
        rv = fwd_al.loc[dt]
        m = fv.notna() & rv.notna()
        if m.sum() < min_obs:
            continue
        valid_dates += 1
        ic = fv[m].corr(rv[m], method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < 100:
        return dict(name=name, n_dates=len(ics), ic=np.nan, icir=np.nan, hit=np.nan)
    ic_mean = ics.mean()
    ic_std = ics.std(ddof=1)
    return dict(name=name, n_dates=len(ics), ic=round(ic_mean, 4), icir=round(ic_mean/ic_std, 4),
                hit=round((ics > 0).mean(), 3), ic_std=round(ic_std, 4))

print("="*100)
rows = []
for name, f in cands.items():
    for h in (1, 5, 10):
        r = eval_factor(name, f, px, h=h)
        rows.append((h, r))
        print(f"h={h:>2} {name:<20} n_dates={r['n_dates']:>5} IC={r['ic']:>7} ICIR={r['icir']:>7} hit={r['hit']:>5} ic_std={r.get('ic_std','-')}")

# coverage snapshot (last 120d)
print("\nCoverage (fraction valid, last 120d):")
for name, f in cands.items():
    cov = f.tail(120).notna().mean().mean()
    print(f"  {name:<20} {cov:.3f}")
