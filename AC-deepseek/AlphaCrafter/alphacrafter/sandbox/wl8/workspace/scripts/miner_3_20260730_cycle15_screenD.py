"""miner_3 cycle-15 screen-D (2026-07-30): fresh orthogonal candidate families.
Fixes batch-C bugs (macro beta alignment, None formatting) and adds new ideas:
  - gold_beta_60        : 60d beta of asset returns to XAU returns (fixed alignment)
  - cn10y_beta_60       : 60d beta of asset returns to CN10Y changes
  - beta_asym_60        : downside beta - upside beta vs equal-weight market
  - ret_autocorr_20     : lag-1 return autocorrelation (20d)
  - volume_z_20         : 20d mean volume z-score vs trailing 60d
  - semi_vol_ratio_20   : downside/upside dev ratio (20d)
  - vol_term_60         : rv20/rv60 ratio
  - skew_60             : 60d rolling return skewness
  - range_pos_20        : close position in 20d high-low range
  - trend_tstat_60      : 60d OLS t-stat of log-price trend (trend consistency, NEW)
  - cvar_60             : 5% daily CVaR over 60d (tail risk, NEW)
  - jump_intensity_60   : share of |ret| > 2*sigma20 days in last 60d (NEW)
  - close_vwap_20       : close / 20d VWAP - 1 (NEW)
  - yield_spread_beta_60: 60d beta to (US10Y-CN10Y) spread changes (NEW)
  - up_vol_share_60     : up-day variance share of 60d total variance (NEW)
Admission h=10: |IC|>=0.007, |ICIR|>=0.084, and max|rho| < 0.5 (BOTH pearson &
spearman) vs the 3 CURRENT library factors.
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
macro["__market_close__"] = close.mean(axis=1)
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}")


def load_lib_artifacts(fids):
    lib = {}
    for fid in fids:
        d = json.load(open(f"factors/{fid}.json"))
        data = d["validation"]["signal_artifact"]["data"]
        raw = base64.b64decode(data)
        csv_text = zlib.decompress(raw).decode()
        panel = pd.read_csv(io.StringIO(csv_text), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib


LIB_FIDS = ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]
lib = load_lib_artifacts(LIB_FIDS)


def lib_corr(panel, method="pearson"):
    out = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            out[fid] = None
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            out[fid] = None
            continue
        if method == "pearson":
            out[fid] = float(np.corrcoef(a[m], b[m])[0, 1])
        else:
            out[fid] = float(spearmanr(a[m], b[m])[0])
    vals = [abs(v) for v in out.values() if v is not None and np.isfinite(v)]
    return out, (max(vals) if vals else None)


# ---- factor functions (dense per-asset calendars) ----
def _beta_to(ri, rx):
    """rolling 60d beta of ri to rx on ri's own index; rx pre-aligned to ri index."""
    rx = rx.reindex(ri.index)
    cov = ri.rolling(60).cov(rx)
    var = rx.rolling(60).var().replace(0, np.nan)
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def f_gold_beta(c, v, o, h, l, m, win=60):
    g = m["XAU_close"]
    ri = c.pct_change()
    rx = g.pct_change()
    return _beta_to(ri, rx)


def f_cn10y_beta(c, v, o, h, l, m, win=60):
    y = m["CN10Y_close"]
    ri = c.pct_change()
    ry = y.pct_change()
    return _beta_to(ri, ry)


def f_yield_spread_beta(c, v, o, h, l, m, win=60):
    us = m["US10Y_close"]
    cn = m["CN10Y_close"]
    sp = (us - cn)  # spread level
    ri = c.pct_change()
    rsp = sp.diff()
    return _beta_to(ri, rsp)


def f_beta_asym(c, v, o, h, l, m, win=60):
    mkt = m["__market_close__"]
    ri = c.pct_change()
    rm = mkt.pct_change()
    df = pd.concat([ri, rm], axis=1).dropna()
    if len(df) < win:
        return pd.Series(np.nan, index=ri.index)
    rb = df.iloc[:, 0]; rm_ = df.iloc[:, 1]
    down = rm_ < 0; up = rm_ >= 0
    bd = _beta_to(rb, rm_.where(down))
    bu = _beta_to(rb, rm_.where(up))
    return (bd - bu).reindex(ri.index)


def f_ret_autocorr(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: x.autocorr() if len(x) > 3 else np.nan, raw=False)


def f_volume_z(c, v, o, h, l, m, win=20, base=60):
    vv = v.replace(0, np.nan)
    mu = vv.rolling(base).mean(); sd = vv.rolling(base).std()
    return ((vv - mu) / sd.replace(0, np.nan)).rolling(win).mean()


def f_semi_vol_ratio(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    dd = r.where(r < 0, np.nan).rolling(win).std()
    ud = r.where(r > 0, np.nan).rolling(win).std()
    return (dd / ud.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_vol_term(c, v, o, h, l, m, s=20, long=60):
    r = c.pct_change()
    vs = r.rolling(s).std(); vl = r.rolling(long).std()
    return (vs / vl.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_skew_60(c, v, o, h, l, m, win=60):
    return c.pct_change().rolling(win).skew()


def f_range_pos_20(c, v, o, h, l, m, win=20):
    hi = h.rolling(win).max(); lo = l.rolling(win).min()
    return ((c - lo) / (hi - lo).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_trend_tstat_60(c, v, o, h, l, m, win=60):
    lp = np.log(c)
    def tstat(x):
        x = x[~np.isnan(x)]
        if len(x) < 30:
            return np.nan
        t = np.arange(len(x))
        A = np.vstack([t, np.ones(len(t))]).T
        try:
            beta, _, _, se, _ = np.linalg.lstsq(A, x, rcond=None)[0], None, None, None, None
            coef, res, rank, sv = np.linalg.lstsq(A, x, rcond=None)
            yhat = A @ coef
            resid = x - yhat
            dof = len(x) - 2
            se_b = np.sqrt((resid @ resid) / dof / np.sum((t - t.mean()) ** 2))
            return coef[0] / se_b if se_b > 0 else np.nan
        except Exception:
            return np.nan
    return lp.rolling(win).apply(tstat, raw=True)


def f_cvar_60(c, v, o, h, l, m, win=60, q=0.05):
    r = c.pct_change()
    def cvar(x):
        x = x[~np.isnan(x)]
        if len(x) < 30:
            return np.nan
        thr = np.quantile(x, q)
        return float(x[x <= thr].mean())
    return r.rolling(win).apply(cvar, raw=True)


def f_jump_intensity_60(c, v, o, h, l, m, win=60, sig=2.0):
    r = c.pct_change()
    sd = r.rolling(20).std()
    return (r.abs() > sig * sd).rolling(win).mean()


def f_close_vwap_20(c, v, o, h, l, m, win=20):
    vv = v.replace(0, np.nan)
    tp = (h + l + c) / 3.0
    vwap = (tp * vv).rolling(win).sum() / vv.rolling(win).sum().replace(0, np.nan)
    return (c / vwap.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)


def f_up_vol_share_60(c, v, o, h, l, m, win=60):
    r = c.pct_change()
    ru = r.where(r > 0, 0.0)
    rd = r.where(r < 0, 0.0)
    up_var = (ru ** 2).rolling(win).sum()
    tot_var = (r ** 2).rolling(win).sum()
    return (up_var / tot_var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# reference series for macro-beta factors (aligned per asset inside factor_panel)
macro["XAU_close"] = close["XAU"].astype(float)
macro["CN10Y_close"] = close["CN10Y"].astype(float)
macro["US10Y_close"] = close["US10Y"].astype(float)

CANDIDATES = [
    ("gold_beta_60", f_gold_beta, "60d beta to XAU returns"),
    ("cn10y_beta_60", f_cn10y_beta, "60d beta to CN10Y yield changes"),
    ("yield_spread_beta_60", f_yield_spread_beta, "60d beta to US10Y-CN10Y spread changes"),
    ("beta_asym_60", f_beta_asym, "downside beta - upside beta (60d)"),
    ("ret_autocorr_20", f_ret_autocorr, "lag-1 return autocorrelation (20d)"),
    ("volume_z_20", f_volume_z, "20d mean volume z-score vs 60d"),
    ("semi_vol_ratio_20", f_semi_vol_ratio, "downside/upside dev ratio (20d)"),
    ("vol_term_60", f_vol_term, "20d/60d close-vol ratio"),
    ("skew_60", f_skew_60, "60d rolling return skewness"),
    ("range_pos_20", f_range_pos_20, "close position in 20d high-low range"),
    ("trend_tstat_60", f_trend_tstat_60, "60d OLS t-stat of log-price trend"),
    ("cvar_60", f_cvar_60, "5% daily CVaR over 60d (tail risk)"),
    ("jump_intensity_60", f_jump_intensity_60, "share of |ret|>2sd days in 60d"),
    ("close_vwap_20", f_close_vwap_20, "close / 20d VWAP - 1"),
    ("up_vol_share_60", f_up_vol_share_60, "up-day variance share of 60d variance"),
]

HORIZONS = (1, 2, 3, 5, 10, 20)
results = {}
for name, fn, desc in CANDIDATES:
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay = {}
    for h in HORIZONS:
        ic = ic_series(panel, fwd_returns(close, h))
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic10 = ic_series(panel, fwd_returns(close, 10))
    ic = float(ic10.mean()) if len(ic10) else np.nan
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if np.isfinite(ic) and ic < 0:
        hit = float((ic10 < 0).mean())
    pe_map, maxpe = lib_corr(panel, "pearson")
    sp_map, maxsp = lib_corr(panel, "spearman")
    ok = (np.isfinite(ic) and abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
          and maxpe is not None and maxsp is not None and maxpe < 0.5 and maxsp < 0.5)
    results[name] = dict(ic=ic, icir=icir, hit=hit, n=len(ic10), cov_ad=cov_ad,
                         cov_ge8=cov_ge8, to=to, decay=decay, pe=pe_map, sp=sp_map,
                         maxpe=maxpe, maxsp=maxsp, ok=ok)
    print(f"\n=== {name} [{desc}] ===")
    print(f"  IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} cov_ad={cov_ad:.3f} "
          f"cov8={cov_ge8:.3f} turnover={to:.3f}")
    print(f"  decay: { {str(k): round(v, 4) if np.isfinite(v) else None for k, v in decay.items()} }")
    print(f"  pearson vs lib: { {k: round(v, 3) if v is not None else None for k, v in pe_map.items()} } max|rho|={maxpe if maxpe is None else round(maxpe,3)}")
    print(f"  spearman vs lib: { {k: round(v, 3) if v is not None else None for k, v in sp_map.items()} } max|rho|={maxsp if maxsp is None else round(maxsp,3)}")
    print(f"  GATE(IC>={IC_GATE}, ICIR>={ICIR_GATE}, rho<0.5): {'PASS' if ok else 'FAIL'}")

print(f"\n===== SUMMARY (h=10) =====  elapsed={time.time()-t0:.1f}s")
for name, r in results.items():
    ic = r['ic'] if np.isfinite(r['ic']) else float('nan')
    print(f"{name:24s} IC={ic:+.4f} ICIR={r['icir'] if np.isfinite(r['icir']) else float('nan'):+.4f} "
          f"hit={r['hit']:.3f} n={r['n']} cov={r['cov_ad']:.3f} to={r['to']:.2f} "
          f"maxpe={r['maxpe'] if r['maxpe'] is None else round(r['maxpe'],3)} "
          f"maxsp={r['maxsp'] if r['maxsp'] is None else round(r['maxsp'],3)} -> {'PASS' if r['ok'] else 'fail'}")
