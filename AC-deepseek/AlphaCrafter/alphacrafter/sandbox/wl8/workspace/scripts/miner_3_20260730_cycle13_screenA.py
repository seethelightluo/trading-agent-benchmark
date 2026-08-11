"""miner_3 cycle-13 screen-A (2026-07-30).
Novel factor families NOT previously tried (checked evicted/rejected library):
  - ret_skew_30        : 30d rolling skewness of daily returns (kurtosis tried, skewness not)
  - sharpe_60          : rolling Sharpe (mean/std of daily returns, 60d)
  - downside_sharpe_60 : mean / downside-dev (Sortino-like)
  - maxdd_60           : 60d max peak-to-trough drawdown (hl_pos was range position, not drawdown)
  - gap_share_20       : overnight gap share of total daily move (overnight_ret_20 was raw drift)
  - overnight_drift_20 : normalized overnight drift (open/prev_close) by 20d close vol
  - coskew_60          : coskewness vs equal-weight cross-asset market (3rd moment)
  - price_impact_60    : corr(|daily ret|, volume) 60d  (pv_corr used signed ret vs log vol)
  - rel_vol_20         : asset 20d vol / cross-sectional median vol (relative amplitude)
Admission gates at h=10: |IC|>=0.0070, |ICIR|>=0.0840; orthogonality rho<0.5 vs library.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE, load_library_panels,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}")
lib = load_library_panels()
print(f"library for orthogonality: {list(lib.keys())}")

HORIZONS = (1, 2, 3, 5, 10, 20)


def f_ret_skew_30(c, v, o, h, l, m, win=30):
    return c.pct_change().rolling(win).skew()


def f_sharpe_60(c, v, o, h, l, m, win=60):
    r = c.pct_change()
    sd = r.rolling(win).std()
    return (r.rolling(win).mean() / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_downside_sharpe_60(c, v, o, h, l, m, win=60):
    r = c.pct_change()
    dd = r.where(r < 0, 0.0).rolling(win).std()
    return (r.rolling(win).mean() / dd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_maxdd_60(c, v, o, h, l, m, win=60):
    roll_max = c.rolling(win).max()
    dd = c / roll_max - 1.0
    return dd.rolling(win).min()  # negative drawdown


def f_gap_share_20(c, v, o, h, l, m, win=20):
    gap = (o / c.shift(1) - 1.0).abs()
    intra = (c / o - 1.0).abs()
    denom = (gap + intra).replace(0, np.nan)
    return (gap / denom).rolling(win).mean()


def f_overnight_drift_20(c, v, o, h, l, m, win=20):
    gap = o / c.shift(1) - 1.0
    sd = c.pct_change().rolling(win).std()
    return (gap.rolling(win).mean() / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_coskew_60(c, v, o, h, l, m, win=60):
    # coskewness of asset vs equal-weight cross-asset market (raw panel close)
    mkt = pd.DataFrame(m["__market_close__"]).iloc[:, 0] if "__market_close__" in m else None
    ri = c.pct_change()
    rm = mkt.pct_change()
    mu_i, mu_m = ri.rolling(win).mean(), rm.rolling(win).mean()
    si, sm = ri.rolling(win).std(), rm.rolling(win).std()
    num = ((ri - mu_i) * (rm - mu_m) ** 2).rolling(win).mean()
    den = (si * sm ** 2).replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def f_price_impact_60(c, v, o, h, l, m, win=60):
    vv = v.replace(0, np.nan)
    return c.pct_change().abs().rolling(win).corr(vv)


def f_rel_vol_20(c, v, o, h, l, m, win=20):
    # cross-sectional median vol computed in factor_panel wrapper via macro dict
    cvol = m["__cs_vol__"]
    avol = c.pct_change().rolling(win).std()
    med = cvol.reindex(c.index).ffill()
    return (avol / med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# ---- cross-sectional helpers passed via macro ----
ew_close = close.mean(axis=1)
cs_vol = close.pct_change().rolling(20).std().median(axis=1)
macro["__market_close__"] = ew_close
macro["__cs_vol__"] = cs_vol

CANDIDATES = [
    ("ret_skew_30", f_ret_skew_30, "30d rolling skewness of daily returns"),
    ("sharpe_60", f_sharpe_60, "60d rolling Sharpe (mean/std daily ret)"),
    ("downside_sharpe_60", f_downside_sharpe_60, "Sortino-like 60d mean/downside-dev"),
    ("maxdd_60", f_maxdd_60, "60d max peak-to-trough drawdown (negative)"),
    ("gap_share_20", f_gap_share_20, "overnight gap share of total daily move, 20d mean"),
    ("overnight_drift_20", f_overnight_drift_20, "normalized overnight drift, 20d"),
    ("coskew_60", f_coskew_60, "60d coskewness vs equal-weight market"),
    ("price_impact_60", f_price_impact_60, "60d corr(|daily ret|, volume)"),
    ("rel_vol_20", f_rel_vol_20, "asset 20d vol / cross-sectional median vol"),
]

results = {}
for name, fn, desc in CANDIDATES:
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay = {}
    for h in HORIZONS:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic10 = ic_series(panel, fwd_returns(close, 10))
    ic = float(ic10.mean())
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic10 < 0).mean())
    # library orthogonality (Spearman to match gate; also Pearson)
    sp_rho = {}
    pe_rho = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            sp_rho[fid] = pe_rho[fid] = np.nan
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        msk = np.isfinite(a) & np.isfinite(b)
        if msk.sum() < 200:
            sp_rho[fid] = pe_rho[fid] = np.nan
            continue
        from scipy.stats import spearmanr
        sp_rho[fid] = float(spearmanr(a[msk], b[msk])[0])
        pe_rho[fid] = float(np.corrcoef(a[msk], b[msk])[0, 1])
    maxsp = max([abs(v) for v in sp_rho.values() if np.isfinite(v)] or [np.nan])
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE and maxsp < 0.5
    results[name] = dict(panel=panel, ic=ic, icir=icir, hit=hit, n=len(ic10),
                         cov_ad=cov_ad, cov_ge8=cov_ge8, to=to, decay=decay,
                         sp_rho=sp_rho, pe_rho=pe_rho, maxsp=maxsp, ok=ok)
    print(f"\n=== {name} [{desc}] ===")
    print(f"  IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} cov_ad={cov_ad:.3f} "
          f"cov8={cov_ge8:.3f} turnover={to:.3f}")
    print(f"  decay: { {str(k): round(v, 4) for k, v in decay.items()} }")
    print(f"  spearman vs lib: { {k: round(v, 3) if np.isfinite(v) else None for k, v in sp_rho.items()} } "
          f"max|rho|={maxsp:.3f}")
    print(f"  GATE(IC>={IC_GATE}, ICIR>={ICIR_GATE}, rho<0.5): {'PASS' if ok else 'FAIL'}")

print(f"\n===== SUMMARY (h=10) =====  elapsed={time.time()-t0:.1f}s")
for name, r in results.items():
    print(f"{name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} hit={r['hit']:.3f} "
          f"n={r['n']} cov={r['cov_ad']:.3f} to={r['to']:.2f} maxsp={r['maxsp']:.3f} -> {'PASS' if r['ok'] else 'fail'}")
