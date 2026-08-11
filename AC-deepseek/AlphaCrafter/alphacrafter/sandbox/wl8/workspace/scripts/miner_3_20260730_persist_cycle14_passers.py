"""miner_3 cycle-14 persistence (2026-07-30).
Persist screen-A passers with full validation records + signal artifacts:
  1. maxdd_60         : 60d max peak-to-trough drawdown (IC<0 -> direction -1)
  2. price_impact_60  : 60d corr(|daily ret|, volume)   (IC>0 -> direction +1)
Both passed |IC|>=0.007, |ICIR|>=0.084 at h=10 and are orthogonal (rho<0.5)
to the current 3-factor library (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).
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
print(f"panel {close.index[0].date()}..{close.index[-1].date()} rows={len(close)} assets={close.shape[1]}")

# ---- library panels: decode artifacts of the 3 CURRENT effective factors ----
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
print(f"library artifacts loaded: {list(lib.keys())}")

def lib_corr(panel):
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
        out[fid] = float(np.corrcoef(a[m], b[m])[0, 1])
    vals = [abs(v) for v in out.values() if v is not None and np.isfinite(v)]
    return out, (max(vals) if vals else None)

def regime_table(fn, params):
    out = {}
    for name, (s, e) in {
        "2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
        "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
        "2024-2026-07 crypto/commodity": ("2024-01-01", None),
    }.items():
        sub_close = close.loc[close.index >= pd.Timestamp(s)] if s else close
        if e:
            sub_close = sub_close.loc[sub_close.index <= pd.Timestamp(e)]
        # rebuild factor on sub-window using same fn (needs full history for warmup; use full panel sliced)
        f_slice = panel.copy()
        if s:
            f_slice = f_slice.loc[f_slice.index >= pd.Timestamp(s)]
        if e:
            f_slice = f_slice.loc[f_slice.index <= pd.Timestamp(e)]
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
          f"art_len={len(back['validation']['signal_artifact']['data'])} reload_ok=True")
    return path

# ============ candidate 1: maxdd_60 ============
def f_maxdd_60(c, v, o, h, l, m, win=60):
    roll_max = c.rolling(win).max()
    return (c / roll_max - 1.0).rolling(win).min()

panel = factor_panel(f_maxdd_60, close, vol, open_, high, low, macro)
fr = fwd_returns(close, 10)
ic10 = ic_series(panel, fr)
ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std())
hit = float((ic10 < 0).mean()) if ic < 0 else float((ic10 > 0).mean())
cov_ad, cov_ge8 = coverage(panel)
to = turnover_rank(panel)
decay = {}
for h in (1, 2, 3, 5, 10, 20):
    s = ic_series(panel, fwd_returns(close, h))
    decay[str(h)] = round(float(s.mean()), 4)
rho_map, maxrho = lib_corr(panel)
reg = regime_table(f_maxdd_60, {})
print("=" * 70)
print(f"maxdd_60: IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} "
      f"cov={cov_ad:.3f} cov8={cov_ge8:.3f} to={to:.2f} maxrho={maxrho}")
print(f"  decay={decay} regime={reg} rho={ {k: round(v,3) if v is not None else None for k,v in rho_map.items()} }")
assert abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE and (maxrho is None or maxrho < 0.5)
rec1 = dict(
    factor_id="maxdd_60", factor_name="60d Max Drawdown",
    calculation=dict(
        expression="(close / rolling_max(close,60) - 1).rolling(60).min()",
        description="Most negative peak-to-trough drawdown over the past 60 trading days. "
                    "Negative values; assets in deeper drawdowns tend to keep underperforming "
                    "(trend continuation) over 10d horizon."),
    dependencies=["close"], parameters={"lookback": 60},
    tags=["drawdown", "trend", "risk", "cross-asset"],
    validation=dict(
        status="EFFECTIVE", period="2020-01-01..2026-07-29", last_validated="2026-07-30",
        admission_horizon=10,
        regime_notes="Validated 2020..2026-07 across COVID crash/recovery, 2022 tightening bear, "
                     "2023-24 AI rally, 2024-26 crypto/commodity cycles. Stable negative IC at "
                     "h=5..20; low turnover (1.44) and high orthogonality to library.",
        metrics=dict(ic=round(ic, 4), icir=round(icir, 4), ic_hit_ratio=round(hit, 4),
                     n_ic_dates=int(len(ic10)), coverage_asset_days=round(cov_ad, 4),
                     coverage_dates_ge8=round(cov_ge8, 4), turnover_10d_rank=round(to, 4),
                     decay_ic_by_horizon=decay, regime_ic_icir=reg,
                     max_abs_library_correlation=round(maxrho, 4) if maxrho else None,
                     library_correlation_detail={k: (round(v, 4) if v is not None else None)
                                                 for k, v in rho_map.items()}),
    ),
)
persist(rec1, panel)

# ============ candidate 2: price_impact_60 ============
def f_price_impact_60(c, v, o, h, l, m, win=60):
    vv = v.replace(0, np.nan)
    return c.pct_change().abs().rolling(win).corr(vv)

panel = factor_panel(f_price_impact_60, close, vol, open_, high, low, macro)
fr = fwd_returns(close, 10)
ic10 = ic_series(panel, fr)
ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std())
hit = float((ic10 < 0).mean()) if ic < 0 else float((ic10 > 0).mean())
cov_ad, cov_ge8 = coverage(panel)
to = turnover_rank(panel)
decay = {}
for h in (1, 2, 3, 5, 10, 20):
    s = ic_series(panel, fwd_returns(close, h))
    decay[str(h)] = round(float(s.mean()), 4)
rho_map, maxrho = lib_corr(panel)
reg = regime_table(f_price_impact_60, {})
print("=" * 70)
print(f"price_impact_60: IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10)} "
      f"cov={cov_ad:.3f} cov8={cov_ge8:.3f} to={to:.2f} maxrho={maxrho}")
print(f"  decay={decay} regime={reg} rho={ {k: round(v,3) if v is not None else None for k,v in rho_map.items()} }")
assert abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE and (maxrho is None or maxrho < 0.5)
rec2 = dict(
    factor_id="price_impact_60", factor_name="60d Price-Impact (|ret|-volume corr)",
    calculation=dict(
        expression="rolling_corr(|pct_change(close)|, volume, 60)",
        description="60-day correlation between absolute daily returns and volume. "
                    "High values indicate volume spikes accompany large moves (impact-heavy "
                    "trading); positively associated with next-10d returns in this cross-asset "
                    "universe. Coverage is reduced because some index series have no volume."),
    dependencies=["close", "volume"], parameters={"lookback": 60},
    tags=["liquidity", "volume", "price-impact", "cross-asset"],
    validation=dict(
        status="EFFECTIVE", period="2020-01-01..2026-07-29", last_validated="2026-07-30",
        admission_horizon=10,
        regime_notes="Validated 2020..2026-07 across multiple regimes; IC positive and growing "
                     "with horizon (0.008 h1 -> 0.055 h20). Coverage asset-days 0.427 due to "
                     "missing/zero volume on some index series; still 1395 valid IC dates with "
                     ">=8 assets. Orthogonal to all library factors (max rho 0.06).",
        metrics=dict(ic=round(ic, 4), icir=round(icir, 4), ic_hit_ratio=round(hit, 4),
                     n_ic_dates=int(len(ic10)), coverage_asset_days=round(cov_ad, 4),
                     coverage_dates_ge8=round(cov_ge8, 4), turnover_10d_rank=round(to, 4),
                     decay_ic_by_horizon=decay, regime_ic_icir=reg,
                     max_abs_library_correlation=round(maxrho, 4) if maxrho else None,
                     library_correlation_detail={k: (round(v, 4) if v is not None else None)
                                                 for k, v in rho_map.items()}),
    ),
)
persist(rec2, panel)

print(f"\n[all done] elapsed={time.time()-t0:.1f}s")
