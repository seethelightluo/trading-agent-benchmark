"""miner_2 2026-07-30 -- persist PASS candidates: body_ratio_20, vol_z_60.
Writes factors/<factor_id>.json with full validation metrics + signal artifact,
then reads back and verifies JSON integrity.
"""
import json
import base64
import zlib
import io
import hashlib
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
    n_total = float(panel.notna().sum().sum())
    cov_ad = n_total / (panel.shape[0] * panel.shape[1])
    cov8 = float((panel.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    ranks = panel.rank(axis=1)
    to = float(ranks.diff(10).abs().mean(axis=1).dropna().mean())
    return {
        "ic": ic, "icir": icir, "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ic10)),
        "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov8, 4),
        "turnover_10d_rank": round(to, 4),
        "decay_ic_by_horizon": {str(h): round(float(out[h].mean()), 4) if len(out[h]) else np.nan for h in HORIZONS},
    }


def artifact(panel):
    csv_text = panel.round(8).to_csv()
    raw = zlib.compress(csv_text.encode(), 9)
    b64 = base64.b64encode(raw).decode()
    return {
        "format": "zlib+base64 csv of factor panel (dates x assets)",
        "description": "Full cross-sectional factor value panel used for admission validation",
        "columns": list(panel.columns),
        "shape": [int(panel.shape[0]), int(panel.shape[1])],
        "n_valid_values": int(panel.notna().sum().sum()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": b64,
    }


def persist(factor_id, factor_name, expression, description, deps, params,
            direction, panel, res, tags):
    res = dict(res)
    res["max_abs_library_correlation"] = round(max_library_corr(panel, lib), 4)
    doc = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "version": "1.0.0",
        "calculation": {"expression": expression, "description": description},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-30",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": ("Validated 2020-01-01..2026-07-30 on the 15-asset cross-asset tradable "
                             "universe across regimes: 2020 COVID crash, 2021 recovery bull, 2022 tightening "
                             "bear, 2023-24 AI equity rally, 2024-26 crypto/commodity cycles."),
            "metrics": res,
            "signal_artifact": artifact(panel),
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                         "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
            "selected_metrics": {"ic": res["ic"], "icir": res["icir"],
                                 "metric_path": "validation.metrics",
                                 "reported_max_abs_library_correlation": res["max_abs_library_correlation"],
                                 "correlation_path": "validation.metrics.max_abs_library_correlation",
                                 "quality": abs(res["ic"]) * abs(res["icir"])},
            "admitted_at": pd.Timestamp.now().isoformat(),
        },
    }
    path = f"factors/{factor_id}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    # read-back verification
    d2 = json.load(open(path))
    assert d2["factor_id"] == factor_id, "id mismatch"
    assert d2["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert abs(d2["validation"]["metrics"]["ic"]) >= IC_GATE, "IC gate"
    assert abs(d2["validation"]["metrics"]["icir"]) >= ICIR_GATE, "ICIR gate"
    assert d2["validation"]["signal_artifact"]["n_valid_values"] > 0, "artifact missing"
    raw2 = base64.b64decode(d2["validation"]["signal_artifact"]["data"])
    assert hashlib.sha256(raw2).hexdigest() == d2["validation"]["signal_artifact"]["sha256"]
    p2 = pd.read_csv(io.StringIO(zlib.decompress(raw2).decode()), index_col=0, parse_dates=True)
    assert p2.shape == panel.shape, "artifact shape mismatch"
    print(f"[OK] {factor_id}: ic={res['ic']:+.4f} icir={res['icir']:+.4f} "
          f"libcorr={res['max_abs_library_correlation']:.4f} artifact {p2.shape} verified")


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


# --- body_ratio_20: mean |body|/range ---
def f_body_ratio_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    return _safe_div((c - o).abs(), rng).rolling(win).mean()


# --- vol_z_60: log-volume z-score ---
def f_vol_z_60(c, v, o, h, l, m, win=60):
    vv = np.log(v.replace(0, np.nan))
    return (vv - vv.rolling(win).mean()) / vv.rolling(win).std()


t0 = time.time()
specs = [
    ("body_ratio_20", "Body/Range Ratio 20d",
     "rolling_mean(|close-open| / (high-low), 20)",
     "20-day mean ratio of absolute candle body to daily high-low range; high values indicate "
     "decisive one-sided sessions (bearish-forward in cross-asset cross-section).",
     ["close", "open", "high", "low"], {"window": 20}, -1, f_body_ratio_20,
     ["structure", "candle", "cross-asset"]),
    ("vol_z_60", "Log-Volume Z-Score 60d",
     "(log(volume) - rolling_mean(log(volume),60)) / rolling_std(log(volume),60)",
     "Z-score of log trading volume over a 60-day window; abnormal volume expansion is "
     "bullish-forward in the cross-asset cross-section.",
     ["volume"], {"window": 60}, +1, f_vol_z_60,
     ["liquidity", "volume", "cross-asset"]),
]
for (fid, fname, expr, desc, deps, params, direction, fn, tags) in specs:
    panel = factor_panel(fn, close, vol, open_, high, low, macro)
    res = fast_validate(panel)
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{fid}: ic={res['ic']:+.4f} icir={res['icir']:+.4f} pass={ok}", flush=True)
    assert ok, f"{fid} does not pass gate, refusing to persist"
    persist(fid, fname, expr, desc, deps, params, direction, panel, res, tags)
print(f"done in {time.time()-t0:.1f}s")
