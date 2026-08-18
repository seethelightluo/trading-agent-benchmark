"""Persist cny_beta_60 factor after deep-dive validation (2030-11-14 cycle)."""
import json
import zlib
import base64
import hashlib
import io
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from miner_1_20301114_common import (
    WATCH, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
    rank_ic_series, summarize_ic, decay_analysis, turnover_10d, coverage_stats,
    regime_split, roll_beta, library_correlation,
    lib_trend_r2, lib_semi_down, lib_mom, lib_dxy_beta, lib_vol_of_vol,
    lib_tuw, lib_tail_ratio, lib_vix_beta_cond, lib_kurt, lib_wti_beta,
)

C = ohlcv_panels()['close']
R = C.pct_change()
USDCNY = macro_panel('USDCNY')
RCNY = USDCNY.pct_change()
fwd10 = C.shift(-10) / C - 1.0

beta = roll_beta(R, RCNY, 60, 30)

ic_s = rank_ic_series(beta, fwd10)
m = summarize_ic(ic_s, 'cny_beta_60')
reg = regime_split(ic_s)
dec = decay_analysis(beta, C)
cov = coverage_stats(beta)
to = turnover_10d(beta)

# ---- robust library correlation (drop non-finite / constant panels) ----
libs = {
    "trend_r2_30_signed": lib_trend_r2(C),
    "semi_down_ratio_20": lib_semi_down(C),
    "mom_120d_skip5": lib_mom(C, 120, 5),
    "dxy_beta_60": lib_dxy_beta(C, macro_panel('DXY')),
    "vol_of_vol20x60": lib_vol_of_vol(C),
    "mom_10d_skip5": lib_mom(C, 10, 5),
    "time_under_water_120": lib_tuw(C),
    "tail_ratio_20": lib_tail_ratio(C),
    "vix_beta_cond_60x20": lib_vix_beta_cond(C, macro_panel('VIX')),
    "kurt_20": lib_kurt(C),
    "WTI_BETA_60": lib_wti_beta(C, C["WTI"]),
}
corrs = {}
for name, lib in libs.items():
    a = beta.stack().replace([np.inf, -np.inf], np.nan)
    b = lib.stack().replace([np.inf, -np.inf], np.nan)
    df = pd.concat([a.rename("f"), b.rename("l")], axis=1).dropna()
    if len(df) < 60 or df["l"].nunique() < 3 or df["f"].nunique() < 3:
        corrs[name] = float("nan")
        continue
    corrs[name] = float(df["f"].corr(df["l"]))
max_abs = max([abs(v) for v in corrs.values() if np.isfinite(v)], default=0.0)
print("library_corr:", {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in corrs.items()})
print("max_abs_library_correlation=%.4f" % max_abs)

# ---- signal artifact ----
panel = beta.copy()
panel.index = panel.index.strftime("%Y-%m-%d")
csv_str = panel.to_csv()
blob = base64.b64encode(zlib.compress(csv_str.encode("utf-8"))).decode("ascii")
sha = hashlib.sha256(blob.encode("ascii")).hexdigest()
n_valid = int(panel.notna().sum().sum())

factor = {
    "factor_id": "cny_beta_60",
    "factor_name": "CNY-beta 60d (USDCNY sensitivity)",
    "version": "1.0.0",
    "calculation": {
        "expression": "beta(asset_ret, USDCNY_ret, 60)",
        "description": "Rolling 60-day beta of each asset's daily returns to the USD/CNY exchange rate (USDCNY): cov(asset_ret, USDCNY_ret, 60)/var(USDCNY_ret, 60), min_periods=30. Captures persistent CNY/USD sensitivity; assets with higher CNY-beta tend to outperform over the 10d horizon (positive IC). Positive CNY-beta implies positive exposure to USD strength vs CNY (or CNY depreciation) - a cross-asset FX-regime tilt."
    },
    "dependencies": ["close", "USDCNY"],
    "parameters": {
        "beta_window": 60,
        "min_periods": 30,
        "horizon": 10,
        "min_valid_assets": 8
    },
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2030-11-13",
        "admission_horizon": 10,
        "last_validated": "2030-11-14",
        "regime_notes": "Validated on 15-asset cross-asset universe, 1765 IC dates with >=8 valid instruments (data through 2030-11-13). Positive IC in every regime bucket: 2020-22 (+0.0429), 2023-24 (+0.0155), 2025-26 (+0.0167), 2027+ (+0.0184), 2028+ (+0.0349), 2029+ (+0.0233), online 2026-07+ (+0.0237); recent last180d IC +0.1161 (ICIR 3.87), last90d IC +0.0893 (ICIR 1.70) - no sign flip. Decay peaks at 3-10d and stays positive at 20d. Low turnover (0.157). Novel vs library: max_abs_library_correlation 0.1314 (vol_of_vol20x60).",
        "metrics": {
            "ic": float(m["ic"]),
            "icir": float(m["icir"]),
            "ic_hit_ratio": float(m["ic_hit_ratio"]),
            "n_ic_dates": int(m["n_ic_dates"]),
            "coverage_asset_days": float(cov["coverage_asset_days"]),
            "coverage_dates_ge8": float(cov["coverage_dates_ge8"]),
            "turnover_10d_rank": float(to),
            "decay_ic_by_horizon": {str(k): float(v) for k, v in dec.items()},
            "max_abs_library_correlation": float(max_abs),
            "library_correlation": {k: (None if not np.isfinite(v) else round(float(v), 6)) for k, v in corrs.items()},
            "regime": {k: {"ic": float(v["ic"]), "icir": float(v["icir"]), "n": int(v["n"])} for k, v in reg.items()}
        },
        "signal_artifact": {
            "format": "base64:zlib:csv",
            "description": "Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. Recover with zlib.decompress(base64.b64decode(data)).decode() -> pandas.read_csv(StringIO).",
            "columns": WATCH,
            "shape": [int(panel.shape[0]), int(panel.shape[1])],
            "n_valid_values": n_valid,
            "sha256": sha,
            "data": blob
        }
    },
    "tags": ["cross_asset", "fx_beta", "macro_regime", "momentum_of_sensitivity"],
    "created": "2030-11-14",
    "last_validated": "2030-11-14"
}

with open("factors/cny_beta_60.json", "w") as f:
    json.dump(factor, f, indent=1)
print("WROTE factors/cny_beta_60.json")

# ---- read-back verification ----
with open("factors/cny_beta_60.json") as f:
    back = json.load(f)
assert back["factor_id"] == "cny_beta_60", "factor_id mismatch"
assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
assert abs(back["validation"]["metrics"]["ic"]) >= 0.0070, "IC gate violated in file"
assert abs(back["validation"]["metrics"]["icir"]) >= 0.0840, "ICIR gate violated in file"
art = back["validation"]["signal_artifact"]
recovered = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
re_panel = pd.read_csv(io.StringIO(recovered), index_col=0)
assert re_panel.shape == tuple(art["shape"]), "shape mismatch: %s vs %s" % (re_panel.shape, art["shape"])
assert hashlib.sha256(art["data"].encode("ascii")).hexdigest() == art["sha256"], "sha256 mismatch"
print("READBACK OK: id=%s status=%s ic=%.4f icir=%.4f shape=%s sha256=%s..." % (
    back["factor_id"], back["validation"]["status"],
    back["validation"]["metrics"]["ic"], back["validation"]["metrics"]["icir"],
    re_panel.shape, art["sha256"][:16]))
print("VERIFIED: factor persisted and reloadable.")
