"""miner_1 cycle-16 (2026-07-30): re-validate carryover passers from cycle-14
batch H (tail_ratio_20, market_corr_60, vol_cluster_60) and explore new
candidate factors on the 15-asset cross-asset universe.

Admission gates (shared benchmark): |IC|>=0.0070, |ICIR|>=0.0840 at h=10,
max abs library correlation < 0.50 vs current 3-factor library.
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    ASSETS, load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE, artifact_b64,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}
macro["__market_close__"] = close.mean(axis=1)
macro["__cs_vol__"] = close.pct_change().rolling(20).std().median(axis=1)
macro["__wti_ret__"] = close["WTI"].pct_change()
macro["__xau_ret__"] = close["XAU"].pct_change()
macro["__usdcny_ret__"] = macro["USDCNY"].pct_change()
macro["__usd_jpy_ret__"] = macro["USDJPY"].pct_change()
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}", flush=True)


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
print(f"library artifacts loaded: {list(lib.keys())}", flush=True)


def lib_corr_datewise(panel):
    """Date-wise mean rank correlation between candidate and library panels."""
    out = {}
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            out[fid] = None
            continue
        ra = panel.loc[common, cols].rank(axis=1)
        rb = lp.loc[common, cols].rank(axis=1)
        rr = []
        for dt in common:
            x, y = ra.loc[dt], rb.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 5:
                rr.append(x[m].corr(y[m], method="pearson"))
        out[fid] = float(np.nanmean(rr)) if rr else np.nan
    vals = [abs(v) for v in out.values() if v is not None and np.isfinite(v)]
    return out, (max(vals) if vals else None)


def lib_corr_ravel(panel):
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
        out[fid] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 200 else None
    vals = [abs(v) for v in out.values() if v is not None and np.isfinite(v)]
    return out, (max(vals) if vals else None)


def regime_table(panel):
    out = {}
    for name, (s, e) in {
        "2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
        "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
        "2024-2026-07 crypto/commodity": ("2024-01-01", None),
    }.items():
        f_slice = panel.loc[panel.index >= pd.Timestamp(s)]
        if e:
            f_slice = f_slice.loc[f_slice.index <= pd.Timestamp(e)]
        sub_close = close.loc[close.index >= pd.Timestamp(s)]
        if e:
            sub_close = sub_close.loc[sub_close.index <= pd.Timestamp(e)]
        ic = ic_series(f_slice, fwd_returns(sub_close, 10))
        out[name] = [round(float(ic.mean()), 4),
                     round(float(ic.mean() / ic.std()), 4) if len(ic) > 2 else None,
                     int(len(ic))]
    return out


def persist(record, panel):
    v = record.setdefault("validation", {})
    v["signal_artifact"] = {
        "format": "base64:zlib:csv",
        "descrip": "factor value panel rows=date cols=asset (15-asset cross-asset universe)",
        "data": artifact_b64(panel),
    }
    record.setdefault("version", "1.0.0")
    m = v["metrics"]
    record["expected_direction"] = 1 if m["ic"] >= 0 else -1
    record["benchmark_admission"] = {
        "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                     "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
        "selected_metrics": {
            "ic": m["ic"], "icir": m["icir"], "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m.get("max_abs_library_correlation"),
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": pd.Timestamp.now().isoformat(),
    }
    path = f"factors/{record['factor_id']}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    with open(path) as f:
        back = json.load(f)
    assert back["factor_id"] == record["factor_id"], "id mismatch"
    assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert "signal_artifact" in back["validation"], "artifact missing"
    assert abs(back["validation"]["metrics"]["ic"]) >= IC_GATE
    assert abs(back["validation"]["metrics"]["icir"]) >= ICIR_GATE
    print(f"[persist] {path} status={back['validation']['status']} "
          f"art_len={len(back['validation']['signal_artifact']['data'])} reload_ok=True", flush=True)
    return path


# ================= factor definitions =================
def tail_ratio_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).max() / r.rolling(win).min().abs()


def market_corr_60(c, v, o, h, l, m, win=60):
    mkt = m["__market_close__"].reindex(c.index).pct_change()
    r = c.pct_change()
    return r.rolling(win).corr(mkt)


def vol_cluster_60(c, v, o, h, l, m, win=60):
    r2 = c.pct_change() ** 2
    return r2.rolling(win).corr(r2.shift(1))


def parkinson_ratio_20(c, v, o, h, l, m, win=20):
    """(log(high/low))^2 mean vs close-close log-return variance: range-vol efficiency."""
    hl = (np.log(h / l) ** 2).rolling(win).mean()
    lr = np.log(c / c.shift(1))
    cc = lr.rolling(win).var()
    return hl / cc


def wti_beta_60(c, v, o, h, l, m, win=60):
    mkt = m["__wti_ret__"].reindex(c.index)
    r = c.pct_change()
    cov = r.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return cov / var


def xau_beta_60(c, v, o, h, l, m, win=60):
    mkt = m["__xau_ret__"].reindex(c.index)
    r = c.pct_change()
    cov = r.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return cov / var


def usdcny_beta_60(c, v, o, h, l, m, win=60):
    mkt = m["__usdcny_ret__"].reindex(c.index)
    r = c.pct_change()
    cov = r.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return cov / var


def gap_tend_20(c, v, o, h, l, m, win=20):
    """Mean overnight gap (open vs prev close): gap continuation tendency."""
    gap = o / c.shift(1) - 1.0
    return gap.rolling(win).mean()


def vol_accel_20x60(c, v, o, h, l, m, win_s=20, win_l=60):
    """Volume acceleration: 20d avg volume / 60d avg volume."""
    vs = v.rolling(win_s).mean()
    vl = v.rolling(win_l).mean()
    return vs / vl


CANDIDATES = [
    ("tail_ratio_20", tail_ratio_20, {}, "20d max-ret / |min-ret| tail asymmetry",
     "20-day ratio of maximum daily return to absolute minimum daily return "
     "(return tail asymmetry). High = positive tail dominates.",
     ["close"], {"window": 20}, ["return-asymmetry", "tail-risk", "cross-asset"]),
    ("market_corr_60", market_corr_60, {}, "60d return corr vs equal-weight market",
     "60d rolling Pearson correlation of asset daily returns with the equal-weight "
     "cross-asset market return (systematic-risk / regime-participation measure).",
     ["close"], {"window": 60}, ["beta", "systematic-risk", "cross-asset"]),
    ("vol_cluster_60", vol_cluster_60, {}, "60d autocorr of squared returns",
     "60d autocorrelation of squared daily returns (volatility clustering).",
     ["close"], {"window": 60}, ["volatility", "garch", "cross-asset"]),
    ("parkinson_ratio_20", parkinson_ratio_20, {}, "range-vol vs close-vol efficiency",
     "20d mean(log(high/low)^2) divided by 20d close-close log-return variance. "
     "High = wide intraday ranges relative to net moves (choppiness/efficiency).",
     ["close", "high", "low"], {"window": 20}, ["volatility", "efficiency", "cross-asset"]),
    ("wti_beta_60", wti_beta_60, {}, "60d beta vs WTI",
     "60d rolling beta of asset daily returns to WTI crude returns "
     "(commodity-cycle participation).",
     ["close"], {"window": 60}, ["beta", "commodity", "cross-asset"]),
    ("xau_beta_60", xau_beta_60, {}, "60d beta vs XAU",
     "60d rolling beta of asset daily returns to gold (XAU) returns "
     "(safe-haven sensitivity).",
     ["close"], {"window": 60}, ["beta", "safe-haven", "cross-asset"]),
    ("usdcny_beta_60", usdcny_beta_60, {}, "60d beta vs USDCNY",
     "60d rolling beta of asset daily returns to USDCNY moves "
     "(China FX / EM exposure).",
     ["close"], {"window": 60}, ["beta", "fx", "cross-asset"]),
    ("gap_tend_20", gap_tend_20, {}, "20d mean overnight gap",
     "20d mean of (open / prev close - 1): overnight gap tendency, "
     "proxy for persistent news-flow / gap continuation behavior.",
     ["close", "open"], {"window": 20}, ["gap", "microstructure", "cross-asset"]),
    ("vol_accel_20x60", vol_accel_20x60, {}, "volume acceleration 20/60",
     "Ratio of 20d average volume to 60d average volume (volume trend / "
     "participation acceleration).",
     ["volume"], {"window_s": 20, "window_l": 60}, ["liquidity", "volume", "cross-asset"]),
]

results = {}
for name, fn, params, fname, desc, deps, pars, tags in CANDIDATES:
    panel = factor_panel(fn, close, vol, open_, high, low, macro, **params)
    fr = fwd_returns(close, 10)
    ic10 = ic_series(panel, fr)
    ic = float(ic10.mean())
    icir = float(ic10.mean() / ic10.std()) if len(ic10) > 2 else np.nan
    hit = float((ic10 > 0).mean()) if np.isfinite(ic) and ic >= 0 else float((ic10 < 0).mean())
    cov_ad, cov_ge8 = coverage(panel)
    to = turnover_rank(panel)
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        s = ic_series(panel, fwd_returns(close, h))
        decay[str(h)] = round(float(s.mean()), 4)
    rho_dw, maxrho_dw = lib_corr_datewise(panel)
    rho_rv, maxrho_rv = lib_corr_ravel(panel)
    maxrho = max([v for v in [maxrho_dw, maxrho_rv] if v is not None], default=None)
    reg = regime_table(panel)
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE and (maxrho is None or maxrho < 0.5)
    print("=" * 70, flush=True)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} "
          f"cov={cov_ad:.3f} cov8={cov_ge8:.3f} to={to:.2f} "
          f"maxrho(dw)={maxrho_dw} maxrho(rav)={maxrho_rv} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  decay={decay} regime={reg}", flush=True)
    print(f"  rho_datewise={ {k: (round(v,3) if v is not None else None) for k,v in rho_dw.items()} }", flush=True)
    results[name] = dict(ic=ic, icir=icir, hit=hit, n=len(ic10), cov_ad=cov_ad,
                         cov8=cov_ge8, to=to, decay=decay, reg=reg,
                         rho_dw=rho_dw, maxrho=maxrho, ok=ok)
    if ok:
        rec = dict(
            factor_id=name, factor_name=fname, version="1.0.0",
            calculation=dict(expression=desc, description=desc),
            dependencies=deps, parameters=pars, tags=tags,
            validation=dict(
                status="EFFECTIVE",
                period=f"{close.index[0].date()}..{close.index[-1].date()}",
                last_validated="2026-07-30",
                admission_horizon=10,
                regime_notes=(
                    f"Validated on full 2020..2026-07 history (n={len(ic10)} IC dates, "
                    f"coverage {cov_ad:.1%} asset-days). Regime IC: {reg}. "
                    f"Decay by horizon: {decay}. Orthogonal to library: max datewise rho={maxrho_dw:.3f}, "
                    f"ravel rho={maxrho_rv:.3f} vs {LIB_FIDS}."),
                metrics=dict(
                    ic=round(ic, 4), icir=round(icir, 4), ic_hit_ratio=round(hit, 4),
                    n_ic_dates=int(len(ic10)), coverage_asset_days=round(cov_ad, 4),
                    coverage_dates_ge8=round(cov_ge8, 4), turnover_10d_rank=round(to, 4),
                    decay_ic_by_horizon=decay, regime_ic_icir=reg,
                    max_abs_library_correlation=round(maxrho, 4) if maxrho else None,
                    library_correlation_detail={k: (round(v, 4) if v is not None else None)
                                                for k, v in rho_dw.items()}),
            ),
        )
        persist(rec, panel)

json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
          open("scripts/_miner1_cycle16_batchI_results.json", "w"), indent=1, default=str)
print(f"\nDONE in {time.time()-t0:.0f}s")
