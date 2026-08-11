"""miner_2 2026-07-30 — persist pv_corr_60 (passed gate, missing from factors/).
Computes the factor panel, embeds the signal artifact, writes factors/pv_corr_60.json,
then verifies by reading the file back.
"""
import sys, json, hashlib, base64, zlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   artifact_b64, IC_GATE, ICIR_GATE)

FID = "pv_corr_60"

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
lib = load_library_panels()


def f_pv_corr_60(c, v, o, h, l, m):
    v = v.replace(0, np.nan)
    r = c.pct_change()
    lv = np.log(v)
    return r.rolling(60).corr(lv)


res = validate_factor(f_pv_corr_60, close, vol, open_, high, low, macro)
res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
ic, icir = res["ic"], res["icir"]
ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
print(f"ic={ic:+.4f} icir={icir:+.4f} hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']} "
      f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
      f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} gate={'PASS' if ok else 'FAIL'}")
assert ok, "factor does not pass gate; do not persist"

panel = res["panel"]
csv_text = panel.to_csv()
raw = zlib.compress(csv_text.encode())
b64 = base64.b64encode(raw).decode()
sha = hashlib.sha256(raw).hexdigest()
n_valid = int(panel.notna().sum().sum())

factor = {
    "factor_id": FID,
    "factor_name": "price-volume correlation 60d",
    "version": "1.0.0",
    "calculation": {
        "expression": "corr(pct_change(close,1), log(volume), 60)",
        "description": "60-day rolling Pearson correlation between daily return and log trading "
                       "volume per asset. Positive when price moves accompany rising participation "
                       "(conviction); negative IC -> flip (low/negative pv-corr assets earn higher "
                       "forward 10d returns).",
    },
    "dependencies": ["close", "volume"],
    "parameters": {"corr_win": 60, "log_volume": True},
    "expected_direction": -1,
    "validation": {
        "status": "EFFECTIVE",
        "period": {"start": "2020-01-01", "end": "2026-07-30"},
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": ("Cross-asset window 2020-01..2026-07 spanning COVID crash, 2022 inflation/rate "
                         "shock, 2023-25 AI rally and crypto cycles. Volume series available for a subset "
                         "of assets (coverage_asset_days=0.43; 58.6% of dates have >=8 valid assets)."),
        "metrics": {
            "ic": round(ic, 4),
            "icir": round(icir, 4),
            "ic_hit_ratio": round(res["ic_hit_ratio"], 4),
            "n_ic_dates": int(res["n_ic_dates"]),
            "coverage_asset_days": res["coverage_asset_days"],
            "coverage_dates_ge8": res["coverage_dates_ge8"],
            "turnover_10d_rank": res["turnover_10d_rank"],
            "decay_ic_by_horizon": res["decay_ic_by_horizon"],
            "max_abs_library_correlation": res["max_abs_library_correlation"],
        },
        "signal_artifact": {
            "format": "base64+zlib+csv",
            "description": "Daily factor panel (dates x 15 assets) used for validation",
            "columns": list(panel.columns),
            "shape": list(panel.shape),
            "n_valid_values": n_valid,
            "sha256": sha,
            "data": b64,
        },
    },
    "tags": ["volume", "liquidity", "price-volume"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": IC_GATE,
            "icir_threshold": ICIR_GATE,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10,
        },
        "selected_metrics": {
            "ic": round(ic, 4),
            "icir": round(icir, 4),
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": res["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
    },
}

with open(f"factors/{FID}.json", "w") as f:
    json.dump(factor, f, indent=1)
print(f"wrote factors/{FID}.json ({len(b64)} b64 chars)")

# ---- verify read-back ----
d = json.load(open(f"factors/{FID}.json"))
assert d["factor_id"] == FID
assert d["validation"]["status"] == "EFFECTIVE"
assert abs(d["validation"]["metrics"]["ic"]) >= IC_GATE
assert abs(d["validation"]["metrics"]["icir"]) >= ICIR_GATE
art = d["validation"]["signal_artifact"]
raw2 = base64.b64decode(art["data"])
assert hashlib.sha256(raw2).hexdigest() == art["sha256"], "artifact sha mismatch"
p2 = pd.read_csv(io.StringIO(zlib.decompress(raw2).decode()), index_col=0, parse_dates=True)
assert p2.shape == tuple(art["shape"]), f"shape mismatch {p2.shape}"
print(f"verify OK: id={d['factor_id']} status={d['validation']['status']} "
      f"ic={d['validation']['metrics']['ic']} icir={d['validation']['metrics']['icir']} "
      f"artifact sha256 ok, shape={p2.shape}, n_valid={int(p2.notna().sum().sum())}")
