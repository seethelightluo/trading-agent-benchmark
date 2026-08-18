"""miner_3 persistence (2026-11-19 cycle): write gate-passing factors with signal artifacts.

Six candidates passed admission (|IC|>=0.0070, |ICIR|>=0.0840 at H=10 on 15-instrument universe,
validation window 2020-01-01..2026-11-04, metrics computed by
miner_3_20261119_screen_trendquality_fast.py).
Artifacts: base64(zlib(csv)) of the signal panel (dates x assets), so the deterministic
post-Miner gate can recompute pairwise rho from real signals.
"""
import json, base64, zlib, hashlib, io, time
import numpy as np
import pandas as pd

VISIBLE = "2026-11-04"
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes, vols = {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    vol = pd.DataFrame(vols)
    return px, vol


px, vol = load_panel(VISIBLE)
ret = px.pct_change()
vix = load_close("VIX", VISIBLE, INDEX_DIR)["close"].astype(float)
vixr = vix.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rsum(x, w):
    return x.rolling(w, min_periods=mp(w)).sum()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)


def trend_eff_20d():
    path = ret.abs().rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)
    return (px / px.shift(20) - 1.0).abs() / path


def low_vol_20d():
    return -rs(ret, 20)


def down_vol_ratio_20x120():
    down = (ret.clip(upper=0) * -1)
    return -(rs(down, 20) / rs(down, 120).replace(0, np.nan))


def beta_vix_60d_neg():
    return -beta_of(ret, vixr, 60)


def vmm_20d():
    return px.pct_change(20) / rs(ret, 20).replace(0, np.nan)


def vol_imb_20d():
    upday = (ret > 0).astype(float)
    up_vol = (vol * upday).rolling(20, min_periods=mp(20)).sum()
    dn_vol = (vol * (1 - upday)).rolling(20, min_periods=mp(20)).sum()
    return (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)


FACTORS = {
    "beta_vix_60d_neg": {
        "factor_name": "Negative 60d beta to VIX (defensive resilience)",
        "expression": "-cov(ret, vix_ret, 60) / var(vix_ret, 60)",
        "description": "Negative of 60-day rolling beta of each asset's daily return to VIX daily return. "
                       "High values = assets that fall (or rise) when VIX falls, i.e. defensive/low-risk-off beta. "
                       "Positive IC means defensive assets outperform over 10d horizon.",
        "dependencies": ["close", "VIX_close"],
        "parameters": {"win": 60, "min_periods": 30},
        "tags": ["defensive", "risk_off", "cross_asset"],
        "calc": beta_vix_60d_neg,
        "metrics": {
            "ic": 0.0696, "icir": 0.1620, "ic_hit_ratio": 0.562, "n_ic_dates": 974,
            "coverage_asset_days": 0.853, "coverage_dates_ge8": 0.942, "turnover_10d_rank": 0.085,
            "max_abs_library_correlation": 0.111,
            "decay_ic_by_horizon": {"1": 0.0216, "2": 0.0278, "3": 0.0301, "5": 0.0356, "10": 0.0696, "20": 0.0925},
        },
        "regime_notes": ("Full-sample 2020-01..2026-11 IC=0.0696 ICIR=0.162. Sub-windows all positive: "
                         "2024+ 0.0401/0.0943, 2025+ 0.0561/0.1314, 2026+ 0.0705/0.1888. Stable across regimes, "
                         "low library correlation (0.111). Lowest turnover of the batch (0.085)."),
    },
    "vol_imb_20d": {
        "factor_name": "20d up/down volume imbalance",
        "expression": "(sum(vol*I(ret>0),20) - sum(vol*I(ret<=0),20)) / sum(vol,20)",
        "description": "Volume-weighted buying pressure over 20 days: fraction of volume on up days minus "
                       "fraction on down days. High values = persistent accumulation (flow proxy).",
        "dependencies": ["close", "volume"],
        "parameters": {"win": 20, "min_periods": 10},
        "tags": ["volume", "flow", "liquidity"],
        "calc": vol_imb_20d,
        "metrics": {
            "ic": 0.0542, "icir": 0.1332, "ic_hit_ratio": 0.535, "n_ic_dates": 750,
            "coverage_asset_days": 0.587, "coverage_dates_ge8": 0.923, "turnover_10d_rank": 0.209,
            "max_abs_library_correlation": 0.598,
            "decay_ic_by_horizon": {"1": 0.0086, "2": 0.0255, "3": 0.0327, "5": 0.0176, "10": 0.0542, "20": 0.0549},
        },
        "regime_notes": ("Full-sample IC=0.0542 ICIR=0.133. Recent windows strengthening: 2024+ 0.0594/0.1535, "
                         "2025+ 0.0637/0.173, 2026+ 0.0874/0.2461. Moderate library correlation (0.598); "
                         "requires volume series (equity indices + crypto well covered)."),
    },
    "vmm_20d": {
        "factor_name": "20d vol-managed momentum",
        "expression": "pct_change(close,20) / std(ret,20)",
        "description": "20-day return scaled by 20-day realized volatility (return-per-risk momentum). "
                       "Rewards smooth trends rather than raw moves.",
        "dependencies": ["close"],
        "parameters": {"win": 20, "min_periods": 10},
        "tags": ["momentum", "volatility_scaled"],
        "calc": vmm_20d,
        "metrics": {
            "ic": 0.0453, "icir": 0.1292, "ic_hit_ratio": 0.551, "n_ic_dates": 690,
            "coverage_asset_days": 0.531, "coverage_dates_ge8": 0.547, "turnover_10d_rank": 0.249,
            "max_abs_library_correlation": 0.594,
            "decay_ic_by_horizon": {"1": 0.0025, "2": 0.0162, "3": 0.0361, "5": 0.0148, "10": 0.0453, "20": 0.0799},
        },
        "regime_notes": ("Full-sample IC=0.0453 ICIR=0.129. Recent windows positive but softer: "
                         "2024+ 0.0183/0.0497, 2025+ 0.0277/0.0743, 2026+ 0.0118/0.0304. "
                         "Coverage limited (~0.53 asset-days) on instruments with thin vol histories."),
    },
    "down_vol_ratio_20x120": {
        "factor_name": "Downside vol ratio 20x120 (negated)",
        "expression": "-std(max(-ret,0),20) / std(max(-ret,0),120)",
        "description": "Negated ratio of short-term to long-term downside volatility. High values = currently "
                       "low downside vol relative to its norm (calm defensive regime).",
        "dependencies": ["close"],
        "parameters": {"short_win": 20, "long_win": 120, "min_periods_short": 10, "min_periods_long": 60},
        "tags": ["defensive", "downside_volatility"],
        "calc": down_vol_ratio_20x120,
        "metrics": {
            "ic": 0.0386, "icir": 0.1089, "ic_hit_ratio": 0.526, "n_ic_dates": 964,
            "coverage_asset_days": 0.843, "coverage_dates_ge8": 0.925, "turnover_10d_rank": 0.210,
            "max_abs_library_correlation": 0.440,
            "decay_ic_by_horizon": {"1": 0.0068, "2": 0.0102, "3": 0.0102, "5": 0.0188, "10": 0.0386, "20": 0.0501},
        },
        "regime_notes": ("Full-sample IC=0.0386 ICIR=0.109. Strong 2024-2025 (0.0366/0.103, 0.0412/0.111) but "
                         "flipped negative in 2026+ (-0.0471/-0.1331) - regime-dependent; monitor drift. "
                         "Admitted on full-sample gate; may be quarantined on recency by ensemble."),
    },
    "trend_eff_20d": {
        "factor_name": "20d Kaufman trend efficiency",
        "expression": "abs(pct_change(close,20)) / sum(abs(ret),20)",
        "description": "Net 20-day move divided by total 20-day path: efficiency of the prevailing trend. "
                       "High values = smooth directional trends that tend to persist.",
        "dependencies": ["close"],
        "parameters": {"win": 20, "min_periods": 10},
        "tags": ["trend_quality", "efficiency"],
        "calc": trend_eff_20d,
        "metrics": {
            "ic": 0.0331, "icir": 0.0987, "ic_hit_ratio": 0.528, "n_ic_dates": 690,
            "coverage_asset_days": 0.531, "coverage_dates_ge8": 0.547, "turnover_10d_rank": 0.311,
            "max_abs_library_correlation": 0.442,
            "decay_ic_by_horizon": {"1": 0.0227, "2": 0.0345, "3": 0.0425, "5": 0.0213, "10": 0.0331, "20": 0.0451},
        },
        "regime_notes": ("Full-sample IC=0.0331 ICIR=0.099. Positive 2024-2025 (0.0063/0.0185, 0.0094/0.0273), "
                         "negative 2026+ (-0.036/-0.111). Trend-persistence premium faded in 2026; "
                         "admitted on full-sample gate, recency weak."),
    },
    "low_vol_20d": {
        "factor_name": "Low 20d realized vol (negated) - risk-premium direction",
        "expression": "-std(ret,20)",
        "description": "Negated 20-day realized volatility. In this cross-asset universe the factor has NEGATIVE "
                       "IC: high values (low vol assets) UNDERPERFORM over 10d - i.e. a risk-on/vol-premium "
                       "signal favoring recently high-vol assets.",
        "dependencies": ["close"],
        "parameters": {"win": 20, "min_periods": 10},
        "tags": ["volatility", "risk_premium"],
        "calc": low_vol_20d,
        "metrics": {
            "ic": -0.0336, "icir": -0.0905, "ic_hit_ratio": 0.480, "n_ic_dates": 1011,
            "coverage_asset_days": 0.890, "coverage_dates_ge8": 0.967, "turnover_10d_rank": 0.108,
            "max_abs_library_correlation": 0.730,
            "decay_ic_by_horizon": {"1": -0.0016, "2": -0.0101, "3": -0.0293, "5": -0.0246, "10": -0.0336, "20": -0.0277},
        },
        "regime_notes": ("Full-sample IC=-0.0336 ICIR=-0.0905 (abs gate pass). Direction is risk-on (high vol wins). "
                         "2025+ slightly positive (0.0093/0.027), 2026+ strongly negative (-0.0425/-0.1539). "
                         "High library correlation (0.730) - likely near-duplicate of vol_of_vol20x60 family."),
    },
}

ORDER = ["beta_vix_60d_neg", "vol_imb_20d", "vmm_20d", "down_vol_ratio_20x120", "trend_eff_20d", "low_vol_20d"]


def encode_artifact(sig):
    csv_txt = sig.round(6).to_csv()
    comp = zlib.compress(csv_txt.encode())
    b64 = base64.b64encode(comp).decode()
    sha = int.from_bytes(hashlib.sha256(csv_txt.encode()).digest()[:8], "big")
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates, cols = assets",
        "columns": list(sig.columns),
        "shape": [sig.shape[0], sig.shape[1]],
        "n_valid_values": int(sig.notna().sum().sum()),
        "sha256": str(sha),
        "data": b64,
    }


t0 = time.time()
for fid in ORDER:
    spec = FACTORS[fid]
    sig = spec["calc"]().reindex(index=px.index, columns=px.columns)
    doc = {
        "factor_id": fid,
        "factor_name": spec["factor_name"],
        "version": "1.0.0",
        "calculation": {
            "expression": spec["expression"],
            "description": spec["description"],
        },
        "dependencies": spec["dependencies"],
        "parameters": spec["parameters"],
        "expected_direction": 1 if spec["metrics"]["ic"] > 0 else -1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-11-04",
            "last_validated": "2026-11-19",
            "admission_horizon": 10,
            "regime_notes": spec["regime_notes"],
            "metrics": spec["metrics"],
            "signal_artifact": encode_artifact(sig),
        },
        "tags": spec["tags"],
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {path}  sig={sig.shape} valid={sig.notna().sum().sum()} ({time.time()-t0:.1f}s)", flush=True)

# --- read-back verification ---
print("\n=== READ-BACK VERIFICATION ===", flush=True)
for fid in ORDER:
    path = f"factors/{fid}.json"
    d = json.load(open(path))
    art = d["validation"]["signal_artifact"]
    txt = zlib.decompress(base64.b64decode(art["data"])).decode()
    df = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
    ok_shape = list(df.shape) == art["shape"]
    ok_status = d["validation"]["status"] == "EFFECTIVE"
    ok_id = d["factor_id"] == fid
    ok_gate = abs(d["validation"]["metrics"]["ic"]) >= 0.0070 and abs(d["validation"]["metrics"]["icir"]) >= 0.0840
    print(f"{fid:<24} id={ok_id} status={ok_status} gate={ok_gate} shape={df.shape}=={art['shape']}:{ok_shape} "
          f"valid={art['n_valid_values']} ic={d['validation']['metrics']['ic']:.4f} "
          f"icir={d['validation']['metrics']['icir']:.4f}", flush=True)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
