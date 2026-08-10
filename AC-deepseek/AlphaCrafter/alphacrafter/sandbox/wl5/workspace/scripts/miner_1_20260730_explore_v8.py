"""miner_1 cycle 2026-07-30 (v8): fast re-validation of strongest candidates + new ideas.

Speedups vs v7:
  - vectorized rank-IC (Spearman via row-wise Pearson on ranks), validated to match pandas.
  - library IC series computed once.
Saves signal panels (CSV) for gate-passing candidates to scripts/_panels/ so the
persistence step can embed recoverable artifacts.
All research restricted to visible window <= 2026-07-29.
"""
import json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             summary_metrics, library_ic_series_map,
                             max_abs_library_corr, regime_split)

VIS = '2026-07-29'
H = 10
t0 = time.time()
close = closes_panel(VIS)
ret = close.pct_change()
lp = np.log(close)
macro = macro_closes(VIS)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS} load={time.time()-t0:.1f}s", flush=True)


def fast_ic_series(factor, fwd_ret, min_valid=8):
    """Spearman rank IC per date, vectorized via ranks (matches pandas spearman)."""
    f_rank = factor.rank(axis=1)
    r_rank = fwd_ret.rank(axis=1)
    fa = f_rank.values
    ra = r_rank.values
    idx = factor.index
    out = {}
    for i in range(len(factor)):
        frow = fa[i]
        rrow = ra[i]
        mask = ~(np.isnan(frow) | np.isnan(rrow))
        n = int(mask.sum())
        if n < min_valid:
            continue
        fv = frow[mask]
        rv = rrow[mask]
        if len(np.unique(fv)) < 3 or len(np.unique(rv)) < 2:
            continue
        fm = fv - fv.mean()
        rm = rv - rv.mean()
        denom = np.sqrt((fm * fm).sum() * (rm * rm).sum())
        if denom <= 1e-15:
            continue
        ic = float((fm * rm).sum() / denom)
        if np.isfinite(ic):
            out[idx[i]] = ic
    return pd.Series(out, dtype=float)


# ---------------- rolling helpers ----------------
def rolling_beta(asset_ret, mkt_ret, win, mp=36):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename('a'), mkt_ret.rename('m')], axis=1).dropna()
        b = pair['a'].rolling(win, min_periods=mp).cov(pair['m']) / pair['m'].rolling(win, min_periods=mp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


def rolling_corr(asset_ret, mkt_ret, win, mp=36):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename('a'), mkt_ret.rename('m')], axis=1).dropna()
        c = pair['a'].rolling(win, min_periods=mp).corr(pair['m'])
        out[a] = c
    return pd.DataFrame(out).reindex(asset_ret.index)


def rolling_trend_components(df, win):
    t = np.arange(len(df))
    st = pd.Series(t, index=df.index)
    s = df.rolling(win, min_periods=1).sum()
    n = df.rolling(win, min_periods=1).count()
    with np.errstate(all='ignore'):
        mean_y = s / n
        st_w = st.rolling(win, min_periods=1).sum() / n
        sty = (df.mul(st, axis=0)).rolling(win, min_periods=1).sum() / n
        sy2 = (df ** 2).rolling(win, min_periods=1).sum() / n
        st2 = (st ** 2).rolling(win, min_periods=1).sum() / n
        cov = sty - mean_y * st_w
        var_t = st2 - st_w ** 2
        var_y = sy2 - mean_y ** 2
        r2 = (cov ** 2) / (var_t * var_y)
        slope_sign = np.sign(cov)
    mp = int(win * 0.6)
    n = n.where(n >= mp)
    r2 = r2.where((var_t > 1e-12) & (var_y > 1e-12))
    return n, r2, slope_sign


def tstat(n, r2, slope_sign):
    ts = np.sqrt((n - 2) * r2 / (1.0 - r2)) * slope_sign
    return ts.where(r2 < 0.999)


# ---------------- candidate signals ----------------
t0 = time.time()
n30, r2_30, sg30 = rolling_trend_components(lp, 30)
t30 = tstat(n30, r2_30, sg30)
n60, r2_60, sg60 = rolling_trend_components(lp, 60)
t60 = tstat(n60, r2_60, sg60)
print(f"trend components: {time.time()-t0:.1f}s", flush=True)

mom10 = lp.diff(10)
mom20 = lp.diff(20)
mom60 = lp.diff(60)
mom120 = lp.diff(120)
vol20 = ret.rolling(20, min_periods=12).std() * np.sqrt(252)
vol60 = ret.rolling(60, min_periods=36).std() * np.sqrt(252)
down = ret.where(ret < 0, 0.0)
downside_dev60 = np.sqrt((down ** 2).rolling(60, min_periods=36).mean()) * np.sqrt(252)
sma60 = close.rolling(60, min_periods=36).mean()
sma120 = close.rolling(120, min_periods=72).mean()
roll_max60 = close.rolling(60, min_periods=36).max()
roll_min20 = close.rolling(20, min_periods=12).min()
roll_max20 = close.rolling(20, min_periods=12).max()

vix = macro['VIX']
vix_ret = vix.pct_change()
dxy_ret = macro['DXY'].pct_change()

t0 = time.time()
beta_btc60 = rolling_beta(ret, ret['BTC'], 60)
beta_vix60 = rolling_beta(ret, vix_ret, 60)
corr_us10_60 = rolling_corr(ret, close['US10Y'].pct_change(), 60)
corr_dxy_60 = rolling_corr(ret, dxy_ret, 60)
riskon = lp[['SPX', 'NDX', 'SOX', 'N225', 'HSI', 'SX5E', '000300.SH', '000688.SH']].mean(axis=1)
riskon_ret20 = riskon.diff(20)
beta_riskon = rolling_beta(ret, riskon.diff(), 60)
print(f"beta/corr families: {time.time()-t0:.1f}s", flush=True)

cands = {
    # re-validation of known strong ideas (as new artifacts if they pass)
    'trend_tstat_30': t30,
    'trend_tstat_60': t60,
    'trend_r2_30_signed': (r2_30 * sg30),
    'risk_on_alpha_20x60': mom20 - beta_riskon.mul(riskon_ret20, axis=0),
    'vix_beta_cond_60x20': -beta_vix60.mul(vix / vix.shift(20) - 1.0, axis=0),
    'mom_10d_skip5': close.shift(5) / close.shift(15) - 1.0,
    'mom_120d_skip5': close.shift(5) / close.shift(125) - 1.0,
    'vol_of_vol20x60': ret.rolling(20, min_periods=12).std().rolling(60, min_periods=36).std(),
    # new: carry / deviation / positioning
    'carry_60d': close / sma60 - 1.0,
    'carry_120d': close / sma120 - 1.0,
    'drawdown_60d': close / roll_max60 - 1.0,
    'stoch_pos_20d': (close - roll_min20) / (roll_max20 - roll_min20),
    # new: risk-adjusted
    'sharpe_63d': ret.rolling(63, min_periods=38).mean() / ret.rolling(63, min_periods=38).std(),
    'downside_share_60': downside_dev60 / vol60,
    'mom20_vol20': mom20 / vol20,
    # new: cross-asset sensitivities (raw forms)
    'btc_beta_60': beta_btc60,
    'vix_beta_60': beta_vix60,
    'us10y_corr_60': corr_us10_60,
    'dxy_corr_60': corr_dxy_60,
}

fr = forward_returns(close, H)
t0 = time.time()
lib = library_ic_series_map(close, h=H)
print(f"library_ic_series_map: {time.time()-t0:.1f}s lib_size={len(lib)}", flush=True)

results = {}
panels_dir = 'scripts/_panels'
for fid, sig in cands.items():
    t0 = time.time()
    ic_s = fast_ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    dt = time.time() - t0
    if m is None:
        print(f"{fid}: INSUFFICIENT ({len(ic_s)} ic dates) [{dt:.1f}s]", flush=True)
        results[fid] = {"gate_pass": False, "reason": "insufficient IC dates",
                        "n_ic_dates": len(ic_s), "valid_entries": int(sig.notna().sum().sum())}
        continue
    m['max_abs_library_correlation'] = max_abs_library_corr(ic_s, lib)
    m['regime'] = regime_split(ic_s)
    gate = abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084
    m['gate_pass'] = bool(gate)
    results[fid] = m
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} max_rho_lib={m['max_abs_library_correlation']} GATE={gate} [{dt:.1f}s]", flush=True)
    print("  decay:", m['decay_ic_by_horizon'], flush=True)
    print("  regimes:", m['regime'], flush=True)
    if gate:
        # stash panel for persistence artifact
        sig.index = sig.index.strftime('%Y-%m-%d')
        sig.to_csv(f"{panels_dir}/{fid}.csv")

with open('scripts/miner_1_20260730_explore_v8_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE saved scripts/miner_1_20260730_explore_v8_results.json")
