"""miner_1 2029-01-25: revalidate library + screen new factor candidates.

Visible through 2029-01-24 (current_date 2029-01-25). Uses vectorized rank IC
machinery on the 15-asset cross-asset universe (>=8 valid assets per date).
Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at horizon 10.
"""
import json
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

spec = importlib.util.spec_from_file_location("vlib", "scripts/miner_1_20281102_fastlib.py")
vlib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlib)

close = vlib.load_close_panel()
ret = close.pct_change()
macro = vlib.load_macro_panel()
fwd = vlib.forward_returns(close, horizons=(1, 2, 3, 5, 10, 20))
print(f"Panel: dates={len(close)} last={close.index[-1].date()} assets={close.shape[1]}", flush=True)

# ---------------- existing library ----------------
def sig_mom10(): return close.shift(5) / close.shift(15) - 1.0
def sig_mom120(): return close.shift(5) / close.shift(125) - 1.0
def sig_vov(): return ret.rolling(20).std().rolling(60).std()
def sig_lowvol(): return -ret.rolling(20).std()
def sig_betavix_neg():
    vixr = macro['VIX'].pct_change()
    beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    return -beta
def sig_vixcond():
    vixr = macro['VIX'].pct_change()
    beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    return -beta * (macro['VIX'] / macro['VIX'].shift(20) - 1.0)
def sig_dvr():
    down = ret.clip(upper=0).rolling(20).std()
    return down / ret.rolling(120).std()
def sig_betacn10():
    r10 = close['CN10Y'].pct_change()
    return ret.rolling(60).cov(r10) / r10.rolling(60).var()
def sig_betachi():
    rhi = close['HSI'].pct_change()
    return ret.rolling(60).cov(rhi) / rhi.rolling(60).var()
def sig_corr10y():
    r10 = close['US10Y'].pct_change()
    return ret.rolling(60).corr(r10)
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    v = ret.rolling(20).std()
    return v.diff(20) / v.rolling(20).mean()
def sig_xaucop():
    cond = ((close['XAU'].pct_change(20) > 0) & (close['COPPER'].pct_change(20) > 0)).astype(float)
    return cond.mul(-1.0, axis=0)
def sig_volbeta():
    v = ret.rolling(20).std()
    spx_v = ret['SPX'].rolling(20).std()
    return v.rolling(60).cov(spx_v) / spx_v.rolling(60).var()
def sig_signewma():
    return (close / close.ewm(span=60).mean() - 1.0).apply(np.sign)

library = {
    'mom_10d_skip5': sig_mom10, 'mom_120d_skip5': sig_mom120,
    'vol_of_vol20x60': sig_vov, 'low_vol_20d': sig_lowvol,
    'beta_vix_60d_neg': sig_betavix_neg, 'vix_beta_cond_60x20': sig_vixcond,
    'down_vol_ratio_20x120': sig_dvr, 'beta_cn10y_60d': sig_betacn10,
    'beta_chi_60d': sig_betachi, 'corr_us10y_60d': sig_corr10y,
    'skew_20d_neg': sig_skew, 'vol_of_vol_chg_20d': sig_vovchg,
    'xau_copper_cond_20d': sig_xaucop, 'vol_beta_spx_60d': sig_volbeta,
    'sign_ewma_60d': sig_signewma,
}

# ---------------- new candidates ----------------
def sig_dxy_beta():
    rd = macro['DXY'].pct_change()
    return ret.rolling(60).cov(rd) / rd.rolling(60).var()
def sig_usdjpy_beta():
    rj = macro['USDJPY'].pct_change()
    return ret.rolling(60).cov(rj) / rj.rolling(60).var()
def sig_eurusd_beta():
    re_ = macro['EURUSD'].pct_change()
    return ret.rolling(60).cov(re_) / re_.rolling(60).var()
def sig_range_pos_10d():
    ll = close.rolling(10).min()  # use close panel for low/high proxy? use real low/high instead
    return None
def sig_zscore_20d():
    return (close - close.rolling(20).mean()) / close.rolling(20).std()
def sig_dist_high_60d():
    return close / close.rolling(60).max() - 1.0
def sig_sharpe_20d():
    return close.pct_change(20) / (ret.rolling(20).std() * np.sqrt(20))
def sig_reversal_5d():
    return -close.pct_change(5)
def sig_mom60():
    return close.shift(5) / close.shift(65) - 1.0
def sig_updown_vol_ratio_20d():
    up = ret.clip(lower=0).rolling(20).std()
    dn = ret.clip(upper=0).rolling(20).std()
    return up / dn
def sig_vix_cond_lowvol():
    base = -ret.rolling(20).std()
    hi = (macro['VIX'] > macro['VIX'].rolling(120).median()).astype(float)
    return base * (0.5 + 0.5 * hi.values)
def sig_dxy_gated_mom20():
    m = close.pct_change(20)
    g = np.sign(macro['DXY'].pct_change(20))
    return m.mul(g, axis=0)
def sig_vix_gated_mom20():
    m = close.pct_change(20)
    g = np.sign(macro['VIX'].pct_change(20))
    return m.mul(g, axis=0)
def sig_vol_ratio_20x120():
    return ret.rolling(20).std() / ret.rolling(120).std()

candidates = {
    'dxy_beta_60d': sig_dxy_beta,
    'usdjpy_beta_60d': sig_usdjpy_beta,
    'eurusd_beta_60d': sig_eurusd_beta,
    'zscore_20d': sig_zscore_20d,
    'dist_high_60d': sig_dist_high_60d,
    'sharpe_20d': sig_sharpe_20d,
    'reversal_5d': sig_reversal_5d,
    'mom_60d_skip5': sig_mom60,
    'up_down_vol_ratio_20d': sig_updown_vol_ratio_20d,
    'vix_cond_lowvol_20d': sig_vix_cond_lowvol,
    'dxy_gated_mom_20d': sig_dxy_gated_mom20,
    'vix_gated_mom_20d': sig_vix_gated_mom20,
    'vol_ratio_20x120': sig_vol_ratio_20x120,
}

def evaluate(name, fn, label):
    try:
        f = fn().reindex(close.index)
        ic, n = vlib.rank_ic_series(f, fwd[10])
        s = vlib.summarize(ic, n, name, fwd=fwd, factor_df=f, label=label)
        s['gate_ic'] = bool(abs(s['ic']) >= vlib.IC_TH)
        s['gate_icir'] = bool(abs(s['icir']) >= vlib.ICIR_TH)
        s['pass_gate'] = s['gate_ic'] and s['gate_icir']
        # regime splits
        for lo, hi, tag in [('2020-01-01', '2023-12-31', 'r2020_23'),
                            ('2024-01-01', '2026-12-31', 'r2024_26'),
                            ('2027-01-01', '2029-01-24', 'r2027_29')]:
            sub = ic[(ic.index >= lo) & (ic.index <= hi)]
            s[f'{tag}_ic'] = float(sub.mean()) if len(sub) else np.nan
            s[f'{tag}_n'] = int(len(sub))
        print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.2f} n={s['n_dates']:4d} "
              f"cov={s['coverage_asset_days']:.2f} PASS={s['pass_gate']} "
              f"2027_29_ic={s.get('r2027_29_ic', float('nan')):+.3f}", flush=True)
        return s
    except Exception as e:
        print(f"{name:24s} ERROR: {type(e).__name__}: {e}", flush=True)
        return {'factor': name, 'error': str(e)}

results = []
print("\n=== EXISTING LIBRARY REVALIDATION (through 2029-01-24) ===", flush=True)
for name, fn in library.items():
    results.append(evaluate(name, fn, 'reval_20290125'))

print("\n=== NEW CANDIDATES ===", flush=True)
for name, fn in candidates.items():
    results.append(evaluate(name, fn, 'new_20290125'))

json.dump(results, open('scripts/miner_1_20290125_screen_batch_results.json', 'w'), indent=1, default=str)
print("\nSaved results. n_dates_total=", len(close.index), flush=True)
