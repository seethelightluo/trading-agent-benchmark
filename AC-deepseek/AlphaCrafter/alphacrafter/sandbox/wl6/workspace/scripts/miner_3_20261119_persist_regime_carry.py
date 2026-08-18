"""miner_3 persistence (2026-11-19 cycle, part 2): persist gate-passing factors from
miner_3_20261119_screen_regime_carry.py.

Passing candidates (|IC|>=0.0070, |ICIR|>=0.0840 at H=10, n>=250, >=8 instruments/date,
window 2020-01-01..2026-11-18, 15-instrument universe):
  1) beta_cn10y_60d   IC=-0.0523 ICIR=-0.1509 n=658  lib_rho=0.125  direction=-1
  2) vol_imb_10d      IC= 0.0543 ICIR= 0.1349 n=769  lib_rho=0.717  direction=+1
     (10d window variant of active vol_imb_20d; higher quality 0.0073 vs 0.0066,
      so the pairwise-correlation gate should keep it over the 20d sibling)

Note: bvix_vixhi_* (IC 0.092/0.212) is a regime-conditioned clone of active beta_vix_60d_neg
(lib_rho=1.0) -> NOT persisted as a new factor; surfaced as regime insight for the screener.
"""
import json, base64, zlib, hashlib, io, time
import numpy as np
import pandas as pd

VISIBLE = "2026-11-18"
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
cn10y = px["CN10Y"]
cn10y_r = cn10y.pct_change()


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    var_m = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return a.rolling(w, min_periods=mp(w, 2)).cov(mdf) / var_m


def beta_cn10y_60d():
    return beta_of(ret, cn10y_r, 60)


def vol_imb_10d():
    upday = (ret > 0).astype(float)
    up_vol = (vol * upday).rolling(10, min_periods=mp(10)).sum()
    dn_vol = (vol * (1 - upday)).rolling(10, min_periods=mp(10)).sum()
    return (up_vol - dn_vol) / (up_vol + dn_vol).replace(0, np.nan)


FACTORS = {
    "beta_cn10y_60d": {
        "factor_name": "60d beta to CN10Y yield changes (negatively predictive)",
        "expression": "cov(ret, cn10y_pct_change, 60) / var(cn10y_pct_change, 60)",
        "description": "Rolling 60-day beta of each asset's daily return to CN10Y yield daily change. "
                       "Full-sample IC is NEGATIVE: assets with high sensitivity to Chinese 10Y yield moves "
                       "underperform over the next 10 trading days (expected_direction=-1). Orthogonal to the "
                       "existing library (max |spearman rho| = 0.125 vs 8 active factors).",
        "dependencies": ["close", "CN10Y_close"],
        "parameters": {"win": 60, "min_periods": 30},
        "tags": ["rate_sensitivity", "carry_proxy", "cross_asset", "china"],
        "calc": beta_cn10y_60d,
        "metrics": {
            "ic": -0.0523, "icir": -0.1509, "ic_hit_ratio": 0.447, "n_ic_dates": 658,
            "coverage_asset_days": 0.656, "coverage_dates_ge8": 0.680, "turnover_10d_rank": 0.129,
            "max_abs_library_correlation": 0.125,
            "decay_ic_by_horizon": {"1": -0.0204, "2": -0.0167, "3": -0.0220, "5": -0.0209, "10": -0.0523, "20": -0.0480},
        },
        "regime_notes": ("Full-sample 2020-01..2026-11-18 IC=-0.0523 ICIR=-0.1509 (direction -1). "
                         "Sign-consistent across horizons (strongest at H=10) and sub-windows: "
                         "2024+ -0.0426/-0.1255, 2025+ -0.0209/-0.0639, 2026+ -0.0109/-0.0358 (magnitude "
                         "fading but sign stable). CN10Y moves on 1276/1381 days through window. "
                         "Lowest library correlation of the batch (0.125) - genuinely orthogonal rate signal."),
    },
    "vol_imb_10d": {
        "factor_name": "10d up/down volume imbalance",
        "expression": "(sum(vol*I(ret>0),10) - sum(vol*I(ret<=0),10)) / sum(vol,10)",
        "description": "Volume-weighted buying pressure over 10 days: fraction of volume on up days minus "
                       "fraction on down days. Faster flow proxy than vol_imb_20d; recent sub-windows "
                       "monotonically improving (2026+ IC 0.1208 / ICIR 0.3523).",
        "dependencies": ["close", "volume"],
        "parameters": {"win": 10, "min_periods": 5},
        "tags": ["volume", "flow", "liquidity"],
        "calc": vol_imb_10d,
        "metrics": {
            "ic": 0.0543, "icir": 0.1349, "ic_hit_ratio": 0.558, "n_ic_dates": 769,
            "coverage_asset_days": 0.583, "coverage_dates_ge8": 0.913, "turnover_10d_rank": 0.313,
            "max_abs_library_correlation": 0.717,
            "decay_ic_by_horizon": {"1": -0.0037, "2": 0.0201, "3": 0.0128, "5": 0.0035, "10": 0.0543, "20": 0.0523},
        },
        "regime_notes": ("Full-sample IC=0.0543 ICIR=0.1349. Recent windows strengthening monotonically: "
                         "2024+ 0.0724/0.1894, 2025+ 0.0957/0.2598, 2026+ 0.1208/0.3523. Correlated with "
                         "active vol_imb_20d (pairwise spearman ~0.7); quality 0.0073 > vol_imb_20d 0.0066, "
                         "so expected to supersede the 20d sibling in the library gate. Requires volume "
                         "series (equity indices + crypto well covered)."),
    },
}

ORDER = ["beta_cn10y_60d", "vol_imb_10d"]


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
            "period": "2020-01-01..2026-11-18",
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
    print(f"{fid:<22} id={ok_id} status={ok_status} gate={ok_gate} shape={df.shape}=={art['shape']}:{ok_shape} "
          f"valid={art['n_valid_values']} ic={d['validation']['metrics']['ic']:.4f} "
          f"icir={d['validation']['metrics']['icir']:.4f} dir={d['expected_direction']}", flush=True)

# --- pair rho vs vol_imb_20d (diagnostic for gate expectation) ---
upday = (ret > 0).astype(float)
up20 = (vol * upday).rolling(20, min_periods=mp(20)).sum()
dn20 = (vol * (1 - upday)).rolling(20, min_periods=mp(20)).sum()
vimb20 = (up20 - dn20) / (up20 + dn20).replace(0, np.nan)
vimb10 = vol_imb_10d()
common = vimb10.index.intersection(vimb20.index)
rho_dates = []
for dt in common:
    a = vimb10.loc[dt].dropna()
    b = vimb20.loc[dt].dropna()
    if len(a) >= 8 and len(b) >= 8:
        rho_dates.append(np.corrcoef(a.rank(), b.rank())[0, 1])
rho_dates = np.array(rho_dates)
print(f"\nvol_imb_10d vs vol_imb_20d: mean |spearman| = {np.nanmean(np.abs(rho_dates)):.3f} "
      f"over {len(rho_dates)} dates (nan mean over dates)", flush=True)
print(f"\ndone in {time.time()-t0:.1f}s", flush=True)
