"""miner_1 factor screening 2026-07-30 (vectorized).
Explore candidate cross-asset factor families on the 15-instrument tradable universe.
Data used strictly <= 2026-07-29 (visible through previous completed trading day).
Methodology: per-date cross-sectional Spearman rank IC of factor vs H-day forward return,
min 8 valid instruments per date; ICIR = mean/std of daily IC; decay over horizons.
"""
import json, base64, zlib, io, sys
import numpy as np
import pandas as pd

END = "2026-07-29"
ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI", "N225",
          "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
H = 10
MIN_VALID = 8

def load_close(assets, end=END):
    closes = {}
    for a in assets:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end]
        closes[a] = df.set_index("date")["close"].astype(float)
    return pd.DataFrame(closes)

def load_index(name, end=END):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end]
    return df.set_index("date")["close"].astype(float)

close = load_close(ASSETS)
vix = load_index("VIX"); dxy = load_index("DXY")
usdjpy = load_index("USDJPY"); eurusd = load_index("EURUSD"); usdcny = load_index("USDCNY")
vixr = vix.pct_change(); dxy_r = dxy.pct_change(); jpy_r = usdjpy.pct_change()

def ret(p, n): return p / p.shift(n) - 1.0
def roll_std(p, n): return p.pct_change().rolling(n).std()
def roll_skew(p, n): return p.pct_change().rolling(n).skew()
def roll_kurt(p, n): return p.pct_change().rolling(n).kurt()
def rolling_beta(y, xr, n):
    yv = y.pct_change()
    xv = xr.reindex(y.index)
    varx = xv.rolling(n).var()
    cov = yv.rolling(n).cov(xv)
    return cov / varx.replace(0, np.nan)

# ---------------- candidate factor definitions ----------------
FACTORS = {
    "rev_3d": lambda: -ret(close, 3),
    "rev_5d": lambda: -ret(close, 5),
    "dist_high_120": lambda: close / close.rolling(120).max() - 1.0,
    "dist_high_60": lambda: close / close.rolling(60).max() - 1.0,
    "mom_30d": lambda: ret(close, 30),
    "mom_rv_60x20": lambda: ret(close, 60) / roll_std(close, 20).clip(lower=1e-6),
    "vol_term_10x60": lambda: roll_std(close, 10) / roll_std(close, 60).clip(lower=1e-9) - 1.0,
    "downside_vol_60": lambda: -close.pct_change().clip(upper=0).rolling(60).std(),
    "skew_60": lambda: roll_skew(close, 60),
    "skew_20": lambda: roll_skew(close, 20),
    "eff_ratio_20": lambda: (close - close.shift(20)).abs() / close.pct_change().abs().rolling(20).sum().clip(lower=1e-9),
    "beta_spx_60": lambda: rolling_beta(close, close["SPX"].pct_change(), 60),
    "dxy_cond_60x20": lambda: -rolling_beta(close, dxy_r, 60) * (dxy / dxy.shift(20) - 1.0).reindex(close.index),
    "vix_cond_mom_20x20": lambda: ret(close, 20) * (vix / vix.shift(20) - 1.0).reindex(close.index),
    "sma_dev_20": lambda: (close - close.rolling(20).mean()) / close.rolling(20).mean(),
    "kurt_60": lambda: roll_kurt(close, 60),
    "jpy_cond_60x20": lambda: -rolling_beta(close, jpy_r, 60) * (usdjpy / usdjpy.shift(20) - 1.0).reindex(close.index),
    "eur_cond_60x20": lambda: -rolling_beta(close, eurusd.pct_change(), 60) * (eurusd / eurusd.shift(20) - 1.0).reindex(close.index),
    "cny_cond_60x20": lambda: -rolling_beta(close, usdcny.pct_change(), 60) * (usdcny / usdcny.shift(20) - 1.0).reindex(close.index),
}

# precompute forward return matrices for horizons
HORIZONS = [1, 3, 5, 10, 20]
FWD = {hh: (close.shift(-hh) / close - 1.0).values for hh in HORIZONS}

def ic_series(fp_ranked, fwd_mat):
    """Vectorized per-date Spearman IC between ranked factor panel and fwd return matrix."""
    m = np.isfinite(fp_ranked) & np.isfinite(fwd_mat)
    n = m.sum(axis=1)
    f = np.where(m, fp_ranked, 0.0); r = np.where(m, fwd_mat, 0.0)
    mf = f.sum(axis=1) / np.maximum(n, 1); mr = r.sum(axis=1) / np.maximum(n, 1)
    fc = f - mf[:, None] * m; rc = r - mr[:, None] * m
    cov = (fc * rc).sum(axis=1)
    vf = (fc * fc).sum(axis=1); vr = (rc * rc).sum(axis=1)
    denom = np.sqrt(vf * vr)
    ics = np.where((n >= MIN_VALID) & (denom > 1e-14), cov / np.where(denom > 1e-14, denom, 1.0), np.nan)
    return ics, n

def validate(fp):
    rk = fp.rank(axis=1).values
    ics, n = ic_series(rk, FWD[H])
    valid = np.isfinite(ics)
    ics_v = ics[valid]
    ic = float(ics_v.mean()); icir = float(ics_v.mean() / ics_v.std(ddof=1)) if len(ics_v) > 2 else np.nan
    hit = float((ics_v > 0).mean())
    total_cells = fp.shape[0] * fp.shape[1]
    cov_asset = float(fp.notna().to_numpy().sum()) / total_cells
    ge8 = float((fp.notna().sum(axis=1) >= MIN_VALID).mean())
    rkf = fp.rank(axis=1)
    turn = float((rkf - rkf.shift(H)).abs().mean().mean())
    decay = {}
    for hh in HORIZONS:
        ics_h, _ = ic_series(rk, FWD[hh])
        v = ics_h[np.isfinite(ics_h)]
        decay[str(hh)] = float(v.mean()) if len(v) else np.nan
    return {"ic": ic, "icir": icir, "ic_hit_ratio": hit,
            "n_ic_dates": int(valid.sum()), "coverage_asset_days": cov_asset,
            "coverage_dates_ge8": ge8, "turnover_10d_rank": turn,
            "decay_ic_by_horizon": decay}

def load_library_panel(fname):
    d = json.load(open(f"factors/{fname}.json"))
    art = d["validation"]["signal_artifact"]
    csv = zlib.decompress(base64.b64decode(art["data"])).decode()
    p = pd.read_csv(io.StringIO(csv), index_col=0)
    p.index = pd.to_datetime(p.index)
    return p

lib_panels = {n: load_library_panel(n) for n in ["mom_10d_skip5", "mom_120d_skip5", "vix_beta_cond_60x20", "vol_of_vol20x60"]}

def lib_corr(fp):
    out = {}
    for n, lp in lib_panels.items():
        lp_r = lp.reindex(fp.index)
        corrs = []
        for c in fp.columns:
            both = fp[c].notna() & lp_r[c].notna()
            if both.sum() >= 200:
                corrs.append(np.corrcoef(fp[c][both], lp_r[c][both])[0, 1])
        out[n] = float(np.max(np.abs(corrs))) if corrs else np.nan
    return out

print(f"Data window: {close.index.min().date()} -> {close.index.max().date()} | panel rows: {len(close)} | universe: {len(ASSETS)} assets", flush=True)
print(f"Admission gate: |IC|>=0.007 and |ICIR|>=0.084, horizon H={H}, min_valid={MIN_VALID}\n", flush=True)

results = {}
for name, fn in FACTORS.items():
    try:
        fp = fn()
        m = validate(fp)
        corr = lib_corr(fp)
        m["max_abs_library_correlation"] = max(corr.values()) if corr else np.nan
        m["lib_corr_detail"] = {k: round(v, 3) for k, v in corr.items()}
        results[name] = m
        gate = "PASS" if (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084) else "fail"
        print(f"[{gate}] {name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:4d} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} maxLibCorr={m['max_abs_library_correlation']:.3f}", flush=True)
        print(f"   decay={ {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} } libCorr={m['lib_corr_detail']}", flush=True)
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}", flush=True)

with open("scripts/_screen_results_20260730.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "lib_corr_detail"} for k, v in results.items()}, f, indent=1, default=str)
print("\nSaved screen results.", flush=True)
