"""miner1 2031-11-07: screen NEW candidate factor families (not in library) on 15-asset cross-section.

Library already holds: nclv_1d/2d/3d/5d, rev_1d/2d/3d/5d, nbody_1d, id_rev_1d, rev_1d_vs,
mom_120d_skip5, vol_of_vol20x60, vix_beta_cond_60x20. This screen tests orthogonal families:
trend efficiency, return autocorrelation, risk-adjusted momentum, range position 20d,
drawdown proximity, skew, volume participation, macro betas, VIX-level conditional reversal,
intraday range, overnight component, Hurst scaling, relative strength.
Admission gate: |IC1|>=0.0070 and |ICIR1|>=0.0840 (daily paper IC). Report h=1/5/10.
"""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache_20311107.pkl')
close = panel['close']; open_px = panel['open']; high = panel['high']; low = panel['low']
vol = panel['vol']; ret = panel['ret']; macro = panel['macro']

FLAT = ['HSI', 'CN10Y']
px = close.drop(columns=[c for c in FLAT if c in close.columns])
rets = ret.drop(columns=[c for c in FLAT if c in ret.columns])
logpx = np.log(px)

cands = {}

# 1) Trend efficiency 60d: R^2 of linear trend on log price
def _trend_eff(x):
    n = len(x)
    if n < 20 or np.isnan(x).any():
        return np.nan
    t = np.arange(n)
    if np.std(x) == 0:
        return np.nan
    c = np.corrcoef(t, x)[0, 1]
    return c * c
cands['trend_eff_60'] = logpx.rolling(60).apply(_trend_eff, raw=True)

# 2) Return autocorrelation lag-1 over 10d (sign: persistence)
def _ar1(x):
    if len(x) < 6 or np.isnan(x).any():
        return np.nan
    a = x[:-1]; b = x[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]
cands['ar1_10'] = rets.rolling(10).apply(_ar1, raw=True)

# 3) Risk-adjusted momentum: 20d return / 20d realized vol
mom20 = px.pct_change(20)
rv20 = rets.rolling(20).std()
cands['mom20_sharpe'] = (mom20 / rv20)

# 4) Range position 20d
lo20 = px.rolling(20).min(); hi20 = px.rolling(20).max()
cands['range_pos_20'] = (px - lo20) / (hi20 - lo20).replace(0, np.nan)

# 5) Drawdown from 252d high (proximity to peak)
cands['dd_252'] = px / px.rolling(252).max() - 1.0

# 6) 20d return skewness (lottery)
cands['skew_20'] = rets.rolling(20).skew()

# 7) Volume participation: 10d avg vol / 60d avg vol
cands['vol_trend_10_60'] = vol.rolling(10).mean() / vol.rolling(60).mean()

# 8) Volume z-score 20d
vol20m = vol.rolling(20).mean(); vol20s = vol.rolling(20).std()
cands['vol_z_20'] = (vol - vol20m) / vol20s.replace(0, np.nan)

# 9) Macro betas (60d rolling beta of asset daily ret on macro daily ret)
macro_ret = macro.pct_change()
def _beta(x, y):
    if len(x) < 20 or np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    vx = np.var(x)
    if vx == 0:
        return np.nan
    return np.cov(x, y)[0, 1] / vx
for ms in ['DXY', 'USDJPY', 'EURUSD', 'USDCNY']:
    mr = macro_ret[ms]
    betas = rets.apply(lambda s: s.rolling(60).corr(mr) * (s.rolling(60).std() / mr.rolling(60).std()))
    betas = betas.reindex(px.columns)
    cands[f'beta_{ms}_60'] = betas

# 10) VIX level conditional reversal: -5d return * (VIX above 60d median)
vix = macro['VIX']
vix_med = vix.rolling(60).median()
rev5 = -(px.pct_change(5))
vix_hi = (vix > vix_med).astype(float)
cands['rev5_vixhi'] = rev5 * vix_hi.reindex(px.index).ffill()

# 11) Intraday range ratio 20d: mean((high-low)/close)
cands['range_ratio_20'] = ((high - low) / close).rolling(20).mean()

# 12) Overnight component 20d: mean(open_t/close_{t-1} - 1)  (gap persistence)
overnight = open_px / close.shift(1) - 1.0
cands['overnight_20'] = overnight.rolling(20).mean()

# 13) Hurst-style scaling: log(vol60/vol10)/log(sqrt(6)) (persistence exponent)
rv10 = rets.rolling(10).std(); rv60 = rets.rolling(60).std()
cands['hurst_10_60'] = np.log(rv60 / rv10) / np.log(np.sqrt(6.0))

# 14) Relative strength: 20d return minus cross-sectional median 20d return
cands['rel_strength_20'] = mom20 - mom20.median(axis=1)

# 15) Drawdown recovery speed: days since 252d high (negative = further from peak)
def _days_since_peak(x):
    if len(x) < 60 or np.isnan(x).any():
        return np.nan
    peak_idx = int(np.argmax(x))
    return float(len(x) - 1 - peak_idx)
cands['days_since_peak_252'] = px.rolling(252).apply(_days_since_peak, raw=True)

# ---------- IC evaluation ----------
def forward_returns(p, h):
    return p.shift(-h) / p - 1.0

def eval_factor(name, f, p, h=1, min_obs=8, min_dates=100):
    fwd = forward_returns(p, h)
    f_al = f.reindex(p.index)
    fwd_al = fwd.reindex(p.index)
    ics = []
    for dt in p.index:
        fv = f_al.loc[dt]; rv = fwd_al.loc[dt]
        m = fv.notna() & rv.notna()
        if m.sum() < min_obs:
            continue
        ic = fv[m].corr(rv[m], method='spearman')
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(name=name, h=h, n_dates=len(ics), ic=np.nan, icir=np.nan, hit=np.nan)
    ic_mean = ics.mean(); ic_std = ics.std(ddof=1)
    return dict(name=name, h=h, n_dates=len(ics), ic=round(float(ic_mean), 4),
                icir=round(float(ic_mean/ic_std), 4) if ic_std > 0 else np.nan,
                hit=round(float((ics > 0).mean()), 3))

print("="*110)
print("FULL-SAMPLE SCREEN (2020-01-01 -> 2031-11-06), flat artifacts excluded")
print("="*110)
results = []
for name, f in cands.items():
    row = {'name': name}
    for h in (1, 5, 10):
        r = eval_factor(name, f, px, h=h)
        row[f'ic{h}'] = r['ic']; row[f'icir{h}'] = r['icir']
        row[f'hit{h}'] = r['hit']; row[f'n{h}'] = r['n_dates']
        if h == 1:
            row['n_dates'] = r['n_dates']
    results.append(row)
    print(f"{name:<22} IC1={row['ic1']:>7.4f} ICIR1={row['icir1']:>7.4f} hit1={row['hit1']:>5.3f} | "
          f"IC5={row['ic5']:>7.4f} ICIR5={row['icir5']:>7.4f} | IC10={row['ic10']:>7.4f} ICIR10={row['icir10']:>7.4f} | n={row['n_dates']}")

# coverage snapshot (last 120d)
print("\nCoverage (fraction valid, last 120d):")
for name, f in cands.items():
    cov = f.tail(120).notna().mean().mean()
    print(f"  {name:<22} {cov:.3f}")

with open('scripts/miner1_20311107_screen_results.json', 'w') as fp:
    json.dump(results, fp, indent=1, default=str)
