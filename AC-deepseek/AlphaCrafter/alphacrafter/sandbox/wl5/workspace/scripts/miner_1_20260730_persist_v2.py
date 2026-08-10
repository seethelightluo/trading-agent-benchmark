"""miner_1: Persist gate-passing factors from explore_v2 with recoverable signal artifacts.

Three candidates passed the shared admission gate (|IC10| >= 0.0070, |ICIR10| >= 0.0840)
in explore_v2: mom_vol_scaled_20x10, breadth_cond_mom_20, skew_vol_comp_20.
Their signal panels were stashed at scripts/_panels/{fid}.csv. Here we embed the
full panel as base64/zlib/CSV artifact (so the deterministic gate can recover the
signal instead of quarantining), write factors/{fid}.json, then read back and verify.
"""
import sys, json, base64, zlib, hashlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
fr = forward_returns(close, H)

FACTORS = {
    "mom_vol_scaled_20x10": {
        "factor_name": "Volatility-scaled momentum 20x10 (quality momentum)",
        "expression": "mom / vol, mom = close.shift(10)/close.shift(30) - 1; vol = std(pct_change, 20)",
        "description": ("20d momentum (with 10d skip to avoid short-term reversal) scaled by "
                        "inverse 20d volatility. Combines trend-following with a quality tilt: "
                        "assets that rise persistently and calmly (low vol) are favored. Positive "
                        "IC at 10d horizon; strongest in 2025-26 regime."),
        "dependencies": ["close"],
        "parameters": {"lookback": 20, "skip": 10, "vol_win": 20, "horizon": 10, "min_valid_assets": 8},
        "expected_direction": 1,
        "regime_notes": ("Validated on 15-asset cross-asset universe; 307 IC dates with >=8 valid "
                         "instruments. IC by regime: 2020-22 +0.0373 (ICIR 0.106), 2023-24 +0.0106 "
                         "(ICIR 0.031, weak), 2025-26 +0.0960 (ICIR 0.259, strong). Positive h=1 "
                         "IC too (0.0350); decay peaks at 10d horizon."),
        "tags": ["momentum", "quality", "volatility-scaled", "cross-asset"],
    },
    "breadth_cond_mom_20": {
        "factor_name": "Market-breadth conditional momentum 20d",
        "expression": "mom * breadth, mom = close/close.shift(20) - 1; breadth = mean(mom > 0) across universe",
        "description": ("20d momentum of each asset multiplied by the cross-sectional fraction of "
                        "assets with positive 20d momentum (market breadth). Strengthens momentum "
                        "signals in broad risk-on tape and dampens them in narrow rallies. Positive "
                        "IC at 10d horizon; regime-dependent (negative in 2023-24, strong otherwise)."),
        "dependencies": ["close"],
        "parameters": {"mom_win": 20, "horizon": 10, "min_valid_assets": 8},
        "expected_direction": 1,
        "regime_notes": ("Validated on 15-asset cross-asset universe; 645 IC dates with >=8 valid "
                         "instruments. IC by regime: 2020-22 +0.0695 (ICIR 0.190), 2023-24 -0.0151 "
                         "(ICIR -0.042, adverse), 2025-26 +0.0714 (ICIR 0.185). Short-horizon (h=1) "
                         "IC is weak/negative (-0.0065); use at 10d horizon only."),
        "tags": ["momentum", "breadth", "market-regime", "cross-asset"],
    },
    "skew_vol_comp_20": {
        "factor_name": "Skew-volatility compression combo 20d",
        "expression": "skew(pct_change, 20) * (std(pct_change,10)/std(pct_change,60))",
        "description": ("Crash-risk composite: 20d return skewness scaled by the 10d/60d volatility "
                        "ratio (short-term vol expansion/compression). Assets with positive skew AND "
                        "short-term vol expansion (volatile up-moves, potential squeeze) tend to "
                        "outperform over 10d. Most robust candidate: positive IC in all three regimes."),
        "dependencies": ["close"],
        "parameters": {"skew_win": 20, "vol_ratio_short": 10, "vol_ratio_long": 60, "horizon": 10, "min_valid_assets": 8},
        "expected_direction": 1,
        "regime_notes": ("Validated on 15-asset cross-asset universe; 567 IC dates with >=8 valid "
                         "instruments. IC by regime: 2020-22 +0.0553 (ICIR 0.160), 2023-24 +0.0423 "
                         "(ICIR 0.131), 2025-26 +0.0663 (ICIR 0.197). Positive h=1 IC (0.0339). "
                         "Best coverage (0.70 asset-days, 0.77 dates with >=8 assets)."),
        "tags": ["skew", "volatility", "crash-risk", "cross-asset"],
    },
}

for fid, cfg in FACTORS.items():
    print(f"\n===== persisting {fid} =====", flush=True)
    panel = pd.read_csv(f"scripts/_panels/{fid}.csv", index_col=0)
    panel.index = pd.to_datetime(panel.index)
    sig = panel.reindex(close.index)

    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"[FAIL] {fid}: insufficient IC dates, skipping")
        continue
    m["regime"] = regime_split(ic)
    gate_ic = abs(m["ic"]) >= 0.0070
    gate_icir = abs(m["icir"] or 0) >= 0.0840
    print(f"  IC10={m['ic']:+.4f} ICIR10={m['icir']:+.3f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank'] if m['turnover_10d_rank'] else float('nan'):.3f}")
    assert gate_ic and gate_icir, f"{fid} does not pass gates: ic={m['ic']}, icir={m['icir']}"

    # --- signal artifact ---
    sig_out = sig.copy()
    sig_out.index = sig_out.index.strftime("%Y-%m-%d")
    csv_txt = sig_out.to_csv()
    raw = csv_txt.encode("utf-8")
    compressed = zlib.compress(raw, 9)
    b64 = base64.b64encode(compressed).decode("ascii")
    artifact = {
        "format": "base64:zlib:csv",
        "description": ("Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. "
                        "Recover with zlib.decompress(base64.b64decode(data)).decode() -> pandas.read_csv(StringIO)."),
        "columns": WATCH,
        "shape": [int(sig_out.shape[0]), int(sig_out.shape[1])],
        "n_valid_values": int(sig_out.notna().sum().sum()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": b64,
    }

    meta = {
        "factor_id": fid,
        "factor_name": cfg["factor_name"],
        "version": "1.0.0",
        "calculation": {
            "expression": cfg["expression"],
            "description": cfg["description"],
        },
        "dependencies": cfg["dependencies"],
        "parameters": cfg["parameters"],
        "expected_direction": cfg["expected_direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": f"2020-01-01..{VIS}",
            "admission_horizon": H,
            "last_validated": "2026-07-30",
            "regime_notes": cfg["regime_notes"],
            "metrics": m,
            "signal_artifact": artifact,
        },
        "tags": cfg["tags"],
    }

    out = f"factors/{fid}.json"
    with open(out, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  written {out} ({len(json.dumps(meta))} bytes)")

    # --- read back & verify ---
    with open(out) as fh:
        back = json.load(fh)
    assert back["factor_id"] == fid
    assert back["validation"]["status"] == "EFFECTIVE"
    assert abs(back["validation"]["metrics"]["ic"]) >= 0.0070
    assert abs(back["validation"]["metrics"]["icir"] or 0) >= 0.0840
    a = back["validation"]["signal_artifact"]
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig_back = pd.read_csv(io.StringIO(dec), index_col=0)
    sig_back.index = pd.to_datetime(sig_back.index)
    assert sig_back.shape == tuple(a["shape"]), (sig_back.shape, a["shape"])
    assert hashlib.sha256(dec.encode("utf-8")).hexdigest() == a["sha256"]
    rec_ic = ic_series(sig_back.reindex(close.index), fr, min_valid=8)
    rec_icir = float(rec_ic.mean() / rec_ic.std(ddof=1)) if len(rec_ic) > 2 and rec_ic.std(ddof=1) > 0 else None
    print(f"  VERIFY ok: shape={sig_back.shape} sha256 ok recovered IC={rec_ic.mean():+.4f} "
          f"ICIR={rec_icir:+.4f} vs persisted IC={m['ic']:+.4f} ICIR={m['icir']:+.4f}")

print("\nALL PERSISTED AND VERIFIED")
