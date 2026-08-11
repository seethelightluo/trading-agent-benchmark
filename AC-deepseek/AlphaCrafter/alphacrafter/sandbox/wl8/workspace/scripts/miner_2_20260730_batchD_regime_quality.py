"""miner_2 2026-07-30 — Batch D: regime-gated momentum, trend quality, vol-mix, cross-asset lead.

Motivation: prior batches covered price/volume/volatility/beta families; most were
evicted for pairwise correlation with yield_beta_cond_60x20 (>0.5) or failed the
IC/ICIR gate. This batch targets genuinely NEW constructions:
  1) mom10_vixgate   - 10d skip-5 momentum x regime switch on VIX vs 60d median (nonlinear)
  2) mom10_volreg    - 10d skip-5 momentum x sign flip when 20d vol > 60d vol (asset vol regime)
  3) mom10_corrgate  - 10d skip-5 momentum x switch on cross-asset median 60d correlation regime
  4) trend_r2_60     - R^2 of linear trend fit over 60d (trend quality/consistency)
  5) parkinson_20    - Parkinson (high-low) vol / close-close vol over 20d (intraday vs overnight mix)
  6) lead_beta_60    - 60d beta of asset return on LAG-1 SPX return (US overnight lead)
  7) spread_beta_cond_60x20 - beta to (US10Y-CN10Y) spread change x 20d spread move
  8) retvol_corr_60  - 60d corr(|daily ret|, log volume): volume-vol interaction
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   ic_series, fwd_returns, print_result,
                                   IC_GATE, ICIR_GATE, factor_panel)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
    "SPX": close["SPX"],
    "US10Y": close["US10Y"],
    "CN10Y": close["CN10Y"],
}
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}")

# --- global regime series (computed on union panel) -------------------------
r = close.pct_change()
# median pairwise 60d correlation across assets (union panel)
corr_regime = pd.Series(np.nan, index=r.index)
for i in range(60, len(r)):
    c = r.iloc[i-59:i+1].corr()
    vals = c.values[np.triu_indices(c.shape[0], k=1)]
    corr_regime.iloc[i] = np.nanmedian(vals)
corr_regime = corr_regime.rolling(5).mean()  # smooth
macro["CORR_REGIME"] = corr_regime
print(f"corr regime: median={corr_regime.median():.3f} q25={corr_regime.quantile(.25):.3f} q75={corr_regime.quantile(.75):.3f}")

vix = macro["VIX"]
vix_med60 = vix.rolling(60).median()
macro["VIX_MED60"] = vix_med60

# --- factor functions -------------------------------------------------------
def _mom10(c, **kw):
    return c.shift(5) / c.shift(15) - 1.0

def f_mom10_vixgate(c, v, o, h, l, m):
    mom = _mom10(c)
    vix = m["VIX"].reindex(c.index)
    med = m["VIX_MED60"].reindex(c.index)
    gate = np.where(vix < med, 1.0, -1.0)
    return mom * pd.Series(gate, index=c.index)

def f_mom10_volreg(c, v, o, h, l, m):
    mom = _mom10(c)
    r = c.pct_change()
    v20 = r.rolling(20).std()
    v60 = r.rolling(60).std()
    gate = np.where(v20 < v60, 1.0, -1.0)
    return mom * pd.Series(gate, index=c.index)

def f_mom10_corrgate(c, v, o, h, l, m):
    mom = _mom10(c)
    cr = m["CORR_REGIME"].reindex(c.index)
    thr = 0.32
    gate = np.where(cr < thr, 1.0, -1.0)
    return mom * pd.Series(gate, index=c.index)

def f_trend_r2_60(c, v, o, h, l, m, win=60):
    x = np.arange(win)
    def _r2(y):
        if len(y) < win or np.any(~np.isfinite(y)):
            return np.nan
        b = np.polyfit(x, y, 1)
        pred = b[0] * x + b[1]
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return c.rolling(win).apply(_r2, raw=True)

def f_parkinson_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    hl = (np.log(h) - np.log(l))
    park = (hl ** 2).rolling(win).mean() / (4 * np.log(2))
    rv = r.rolling(win).mean() ** 2 + r.rolling(win).var()
    return (park / rv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

def _beta(x, y, win):
    return x.rolling(win).cov(y) / y.rolling(win).var().replace(0, np.nan)

def f_lead_beta_60(c, v, o, h, l, m, win=60):
    spx = m["SPX"].reindex(c.index).pct_change().shift(1)  # lag-1 SPX return
    return _beta(c.pct_change(), spx, win)

def f_spread_beta_cond(c, v, o, h, l, m, win=60):
    spread = (m["US10Y"].reindex(c.index) - m["CN10Y"].reindex(c.index))
    dsp = spread.diff()
    b = _beta(c.pct_change(), dsp, win)
    return b * (spread / spread.shift(20) - 1.0)

def f_retvol_corr_60(c, v, o, h, l, m, win=60):
    vv = v.replace(0, np.nan)
    ar = c.pct_change().abs()
    lv = np.log(vv)
    return ar.rolling(win).corr(lv)

cands = [
    ("mom10_vixgate", f_mom10_vixgate, "10d mom x VIX<median gate"),
    ("mom10_volreg", f_mom10_volreg, "10d mom x vol20<vol60 gate"),
    ("mom10_corrgate", f_mom10_corrgate, "10d mom x low-corr-regime gate"),
    ("trend_r2_60", f_trend_r2_60, "R2 of 60d linear trend"),
    ("parkinson_20", f_parkinson_20, "parkinson vol / realized vol 20d"),
    ("lead_beta_60", f_lead_beta_60, "60d beta on lag-1 SPX ret"),
    ("spread_beta_cond_60x20", f_spread_beta_cond, "beta to yield-spread change x 20d spread move"),
    ("retvol_corr_60", f_retvol_corr_60, "60d corr(|ret|, log vol)"),
]

results = {}
for name, fn, desc in cands:
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["panel"] = panel
    res["max_abs_library_correlation"] = round(max_library_corr(panel, lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    if ok:
        ic10 = ic_series(panel, fwd_returns(close, 10))
        print("  regime IC (h=10):", flush=True)
        for rname, (a, b) in [("2020", ("2020-01-01", "2021-12-31")),
                              ("2022", ("2022-01-01", "2023-12-31")),
                              ("2024", ("2024-01-01", "2025-06-30")),
                              ("2025H2+", ("2025-07-01", "2026-07-30"))]:
            sub = ic10.loc[(ic10.index >= a) & (ic10.index <= b)]
            if len(sub):
                print(f"    {rname}: ic={sub.mean():.4f} n={len(sub)}", flush=True)

print("\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:26s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} "
          f"decay10={res['decay_ic_by_horizon']['10']:+.4f} -> {'PASS' if ok else 'fail'}")
print("done")
