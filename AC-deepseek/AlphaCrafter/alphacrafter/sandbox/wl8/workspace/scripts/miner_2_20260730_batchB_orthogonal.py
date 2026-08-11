"""miner_2 2026-07-30 -- BATCH B screening: new factor families orthogonal to library.
Families: short-term reversal, serial autocorrelation, vol term structure,
downside risk, price-vs-VWAP, kurtosis, overnight gap share, beta change,
market correlation, conditional structure. Admission horizon 10.
Gate: |IC|>=0.0070, |ICIR|>=0.0840, libcorr<0.5.
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   load_library_panels, max_library_corr,
                                   IC_GATE, ICIR_GATE, MIN_ASSETS_PER_DATE)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
for anchor in ["SPX", "XAU", "BTC", "WTI", "NDX", "US10Y"]:
    macro[anchor] = close[anchor].dropna()
lib = load_library_panels()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows; lib={list(lib.keys())}", flush=True)

HORIZONS = [1, 2, 3, 5, 10, 20]


def fwd_ret_panel(horizon):
    out = {}
    for a in close.columns:
        c = close[a].dropna()
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)


fwd = {h: fwd_ret_panel(h) for h in HORIZONS}
fwd_rank = {h: fwd[h].rank(axis=1) for h in HORIZONS}


def fast_validate(panel):
    pr = panel.rank(axis=1)
    out = {}
    for h in HORIZONS:
        fr = fwd_rank[h]
        ics = []
        for dt in panel.index:
            x = pr.loc[dt].values
            y = fr.loc[dt].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= MIN_ASSETS_PER_DATE:
                xv, yv = x[m], y[m]
                if xv.std() == 0 or yv.std() == 0:
                    continue
                ics.append(float(np.corrcoef(xv, yv)[0, 1]))
        out[h] = np.array(ics)
    ic10 = out[10]
    ic = float(ic10.mean()) if len(ic10) else np.nan
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic10 < 0).mean())
    cov_ad = float(panel.notna().sum().sum()) / (panel.shape[0] * panel.shape[1])
    cov8 = float((panel.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    to = float(panel.rank(axis=1).diff(10).abs().mean(axis=1).dropna().mean())
    return {
        "ic": ic, "icir": icir, "ic_hit_ratio": hit, "n_ic_dates": int(len(ic10)),
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov8, 4),
        "turnover_10d_rank": round(to, 4),
        "decay_ic_by_horizon": {str(h): round(float(out[h].mean()), 4) if len(out[h]) else np.nan for h in HORIZONS},
    }


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def f_rev_1d(c, v, o, h, l, m):
    return -(c.pct_change())


def f_rev_5d(c, v, o, h, l, m):
    return -(c / c.shift(5) - 1.0)


def f_autocorr_10(c, v, o, h, l, m, win=10):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 and x.std() > 0 else np.nan, raw=True)


def f_autocorr_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 and x.std() > 0 else np.nan, raw=True)


def f_vol_term_10x60(c, v, o, h, l, m):
    vv = c.pct_change().rolling(10).std()
    vl = c.pct_change().rolling(60).std()
    return _safe_div(vv, vl)


def f_downside_ratio_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    tot = r.rolling(win).std()
    d = r.clip(upper=0).rolling(win).std()
    return _safe_div(d, tot)


def f_price_vwap_20(c, v, o, h, l, m, win=20):
    tp = (h + l + c) / 3.0
    vv = v.replace(0, np.nan)
    vwap = (tp * vv).rolling(win).sum() / vv.rolling(win).sum()
    return _safe_div(c, vwap) - 1.0


def f_kurt_20(c, v, o, h, l, m, win=20):
    return c.pct_change().rolling(win).kurt()


def f_gap_share_20(c, v, o, h, l, m, win=20):
    ogn = _safe_div(o, c.shift(1)) - 1.0
    intra = _safe_div(c, o) - 1.0
    tot = ogn + intra
    return _safe_div(ogn, tot).rolling(win).mean()


def f_range_20(c, v, o, h, l, m, win=20):
    return _safe_div(h - l, c).rolling(win).mean()


def f_mkt_corr_20(c, v, o, h, l, m, win=20):
    spx = m["SPX"].reindex(c.index).ffill().pct_change()
    return c.pct_change().rolling(win).corr(spx)


def f_beta_chg_60x120(c, v, o, h, l, m):
    spx = m["SPX"].reindex(c.index).ffill().pct_change()
    r = c.pct_change()
    b60 = r.rolling(60).cov(spx) / spx.rolling(60).var()
    b120 = r.rolling(120).cov(spx) / spx.rolling(120).var()
    return b60 - b120


def f_body_ratio_10(c, v, o, h, l, m, win=10):
    rng = (h - l).replace(0, np.nan)
    return _safe_div((c - o).abs(), rng).rolling(win).mean()


def f_body_ratio_60(c, v, o, h, l, m, win=60):
    rng = (h - l).replace(0, np.nan)
    return _safe_div((c - o).abs(), rng).rolling(win).mean()


def f_vol_z_20(c, v, o, h, l, m, win=20):
    vv = np.log(v.replace(0, np.nan))
    return (vv - vv.rolling(win).mean()) / vv.rolling(win).std()


def f_vol_z_120(c, v, o, h, l, m, win=120):
    vv = np.log(v.replace(0, np.nan))
    return (vv - vv.rolling(win).mean()) / vv.rolling(win).std()


def f_updown_freq_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: (x > 0).sum() / max((x != 0).sum(), 1), raw=True)


def f_cond_body_mom(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    body = _safe_div((c - o).abs(), rng).rolling(win).mean()
    mom = c / c.shift(win) - 1.0
    return body * np.sign(mom)


cands = [
    ("rev_1d", f_rev_1d, "-1d return (short-term reversal)"),
    ("rev_5d", f_rev_5d, "-5d return (reversal)"),
    ("autocorr_10", f_autocorr_10, "10d return autocorrelation"),
    ("autocorr_20", f_autocorr_20, "20d return autocorrelation"),
    ("vol_term_10x60", f_vol_term_10x60, "10d/60d vol term structure"),
    ("downside_ratio_20", f_downside_ratio_20, "downside vol / total vol 20d"),
    ("price_vwap_20", f_price_vwap_20, "close vs 20d VWAP"),
    ("kurt_20", f_kurt_20, "20d return kurtosis"),
    ("gap_share_20", f_gap_share_20, "20d mean overnight share of daily return"),
    ("range_20", f_range_20, "20d mean (high-low)/close"),
    ("mkt_corr_20", f_mkt_corr_20, "20d correlation to SPX"),
    ("beta_chg_60x120", f_beta_chg_60x120, "60d beta - 120d beta to SPX"),
    ("body_ratio_10", f_body_ratio_10, "10d body/range"),
    ("body_ratio_60", f_body_ratio_60, "60d body/range"),
    ("vol_z_20", f_vol_z_20, "log-vol z-score 20d"),
    ("vol_z_120", f_vol_z_120, "log-vol z-score 120d"),
    ("updown_freq_20", f_updown_freq_20, "20d fraction of up days"),
    ("cond_body_mom", f_cond_body_mom, "body ratio x sign(20d mom)"),
]

results = {}
t0 = time.time()
for i, (name, fn, desc) in enumerate(cands):
    t1 = time.time()
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    res = fast_validate(panel)
    res["max_abs_library_correlation"] = round(max_library_corr(panel, lib), 4)
    results[name] = res
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-HI-CORR" if ok else "fail")
    print(f"[{i+1}/{len(cands)}] {name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} "
          f"hit={res['ic_hit_ratio']:.3f} cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} "
          f"decay10={res['decay_ic_by_horizon']['10']:+.4f} -> {flag} ({time.time()-t1:.1f}s)", flush=True)

print(f"\n===== BATCH B SUMMARY (gate |IC|>=%.4f |ICIR|>=%.4f libcorr<0.5) total %.1fs =====" % (IC_GATE, ICIR_GATE, time.time() - t0))
for name, res in sorted(results.items()):
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-HI-CORR" if ok else "fail")
    dec = res["decay_ic_by_horizon"]
    print(f"{name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} "
          f"decay{{1,5,10,20}}={dec['1']:+.3f}/{dec['5']:+.3f}/{dec['10']:+.3f}/{dec['20']:+.3f} -> {flag}")
