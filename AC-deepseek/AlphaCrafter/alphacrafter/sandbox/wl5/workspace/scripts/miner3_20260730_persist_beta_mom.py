"""miner_3 2026-07-30: persist gate-passing factors with recoverable signal artifacts.

Candidates (validated at h=10, min_valid=8 on the 15-asset universe):
  - ETH_BETA_60  : IC=0.0585, ICIR=0.1379
  - WTI_BETA_60  : IC=0.0346, ICIR=0.0926
  - MOM_REL_EQ_20: IC=0.0387, ICIR=0.1042
NDX_BETA_60 (IC=0.0504, ICIR=0.1156) also passes but has IC-series rho=0.853 with
ETH_BETA_60 (>0.5 conflict threshold) -> intentionally NOT persisted to avoid redundancy.

Each persisted JSON embeds the full signal panel as base64:zlib:csv so the
deterministic post-Miner gate can recover the artifact and recompute pairwise rho.
"""
import sys, json, os, base64, zlib, hashlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)


def rolling_beta(a_ret, m_ret, win, minp=40):
    out = {}
    for a in a_ret.columns:
        pair = pd.concat([a_ret[a].rename("a"), m_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=minp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(a_ret.index)


def make_artifact(sig):
    csv = sig.reset_index().to_csv(index=False)
    raw = zlib.compress(csv.encode("utf-8"), 9)
    b64 = base64.b64encode(raw).decode("ascii")
    sha = hashlib.sha256(raw).hexdigest()
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. Recover with zlib.decompress(base64.b64decode(data)).decode() -> pandas.read_csv(StringIO).",
        "columns": list(sig.columns),
        "shape": list(sig.shape),
        "n_valid_values": int(sig.notna().sum().sum()),
        "sha256": sha,
        "data": b64,
    }


def build_factor(fid, meta, sig):
    ics = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ics, sig, fr, close, h=H)
    reg = regime_split(ics)
    doc = {
        "factor_id": fid,
        "factor_name": meta["factor_name"],
        "version": "1.0.0",
        "calculation": {
            "expression": meta["expression"],
            "description": meta["description"],
        },
        "dependencies": meta["dependencies"],
        "parameters": meta["parameters"],
        "expected_direction": meta.get("expected_direction", 1),
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-29",
            "admission_horizon": H,
            "last_validated": "2026-07-30",
            "regime_notes": meta["regime_notes"],
            "metrics": {
                "ic": m["ic"],
                "icir": m["icir"],
                "ic_hit_ratio": m["ic_hit_ratio"],
                "n_ic_dates": m["n_ic_dates"],
                "coverage_asset_days": m["coverage_asset_days"],
                "coverage_dates_ge8": m["coverage_dates_ge8"],
                "turnover_10d_rank": m["turnover_10d_rank"],
                "decay_ic_by_horizon": m["decay_ic_by_horizon"],
                "max_abs_library_correlation": 0.0,  # effective library (factors/) is empty this cycle
                "regime": reg,
            },
            "signal_artifact": make_artifact(sig),
        },
        "tags": meta["tags"],
    }
    return doc


ETH = build_factor("ETH_BETA_60", {
    "factor_name": "ETH-beta 60d (crypto-risk sensitivity)",
    "expression": "beta(asset_ret, ETH_ret, 60)",
    "description": ("Rolling 60-day beta of each asset's daily returns to Ethereum returns. "
                    "Captures persistent crypto-risk sensitivity across the cross-asset universe; "
                    "assets with higher ETH-beta tend to outperform over the 10d horizon."),
    "dependencies": ["close"],
    "parameters": {"beta_window": 60, "horizon": 10, "min_valid_assets": 8},
    "expected_direction": 1,
    "regime_notes": ("Validated on 15-asset cross-asset universe, 629 IC dates with >=8 valid instruments. "
                     "IC by regime: 2020-22 +0.1027 (ICIR 0.233), 2023-24 +0.0644 (ICIR 0.166), "
                     "2025-26 -0.0290 (ICIR -0.068, weakened). Decay increases with horizon (20d IC 0.0708); "
                     "strongest in risk-off 2020-22 crypto cycle."),
    "tags": ["beta", "crypto", "cross-asset", "risk"],
}, rolling_beta(ret, ret["ETH"], 60))

WTI = build_factor("WTI_BETA_60", {
    "factor_name": "WTI-beta 60d (energy sensitivity)",
    "expression": "beta(asset_ret, WTI_ret, 60)",
    "description": ("Rolling 60-day beta of each asset's daily returns to WTI crude returns. "
                    "Captures persistent energy-sensitivity; energy-linked and inflation-hedge assets "
                    "with higher WTI-beta tend to outperform over the 10d horizon."),
    "dependencies": ["close"],
    "parameters": {"beta_window": 60, "horizon": 10, "min_valid_assets": 8},
    "expected_direction": 1,
    "regime_notes": ("Validated on 15-asset cross-asset universe, 629 IC dates with >=8 valid instruments. "
                     "IC by regime: 2020-22 +0.1062 (ICIR 0.294, strong), 2023-24 -0.0128 (ICIR -0.036), "
                     "2025-26 -0.0325 (ICIR -0.082). Predictive power concentrated in the 2020-22 energy "
                     "cycle; monitor for regime re-emergence."),
    "tags": ["beta", "energy", "cross-asset", "risk"],
}, rolling_beta(ret, ret["WTI"], 60))

MOM = build_factor("MOM_REL_EQ_20", {
    "factor_name": "Equity-relative momentum 20d (cross-sectional demeaned)",
    "expression": "mom20 - mean(mom20) across universe, mom20 = close/close.shift(20) - 1",
    "description": ("20-day momentum of each asset demeaned cross-sectionally against the 15-asset "
                    "universe average (relative strength). Assets with above-average 20d trend "
                    "outperform over the 10d horizon; regime-dependent but re-emerged strongly in 2025-26."),
    "dependencies": ["close"],
    "parameters": {"lookback": 20, "horizon": 10, "min_valid_assets": 8},
    "expected_direction": 1,
    "regime_notes": ("Validated on 15-asset cross-asset universe, 653 IC dates with >=8 valid instruments. "
                     "IC by regime: 2020-22 +0.0655 (ICIR 0.178), 2023-24 -0.0246 (ICIR -0.068, weak), "
                     "2025-26 +0.0715 (ICIR 0.185, resurgent). Decay peaks at 20d horizon; turnover 0.246."),
    "tags": ["momentum", "relative-strength", "cross-asset", "trend"],
}, (close / close.shift(20) - 1.0).sub((close / close.shift(20) - 1.0).mean(axis=1), axis=0))

for doc in (ETH, WTI, MOM):
    fp = f"factors/{doc['factor_id']}.json"
    with open(fp, "w") as f:
        json.dump(doc, f, indent=1)
    print("WROTE", fp)

# ---- read back & verify ----
print("\n=== VERIFY ===")
for fid in ("ETH_BETA_60", "WTI_BETA_60", "MOM_REL_EQ_20"):
    fp = f"factors/{fid}.json"
    d = json.load(open(fp))
    art = d["validation"]["signal_artifact"]
    dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
    s = pd.read_csv(io.StringIO(dec), index_col=0)
    s.index = pd.to_datetime(s.index)
    ok = (d["factor_id"] == fid
          and d["validation"]["status"] == "EFFECTIVE"
          and abs(d["validation"]["metrics"]["ic"]) >= 0.007
          and abs(d["validation"]["metrics"]["icir"]) >= 0.084
          and s.shape == tuple(art["shape"])
          and hashlib.sha256(zlib.compress(s.reset_index().to_csv(index=False).encode("utf-8"), 9)).hexdigest() == art["sha256"])
    print(f"  {fid}: valid_json={True} id_ok={d['factor_id']==fid} status={d['validation']['status']} "
          f"ic={d['validation']['metrics']['ic']} icir={d['validation']['metrics']['icir']} "
          f"artifact_shape={s.shape} artifact_ok={ok}")
