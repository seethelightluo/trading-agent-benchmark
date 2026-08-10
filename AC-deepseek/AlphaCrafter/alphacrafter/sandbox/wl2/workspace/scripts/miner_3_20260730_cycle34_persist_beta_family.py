"""miner_3 cycle 34: persist cycle-33 gate-passing candidates + explore orthogonal axes.

Lessons applied:
- Gate quarantines factors without recoverable signal artifact -> persist calmness_20-style
  schema: signal_artifact = "<id>.signal.npy" string + .npy matrix file + artifact_provenance.
- Gate evicted gain_loss_20 at stacked-Spearman rho 0.604 vs mom20_volproxy60 even though
  per-date cross-sectional corr was ~0.17. So max_abs_library_correlation must be computed
  as STACKED (date x asset) Spearman rho vs all active library artifacts, and kept < 0.5.
- Active library (top-level EFFECTIVE json + .signal.npy): calmness_20, dxy_beta_cond_60x20,
  intraday_drift_20, mom20_volproxy60, usdjpy_beta_cond_120x60.

Candidates:
  A. cycle-33 passers: downbeta_spx_60, lagbeta_spx_60, vol_price_corr_60
  B. new orthogonal axes: overnight_gap_20 (complement of intraday_drift), hilo_pos_20,
     rsi_14, kurtosis_60, tail_ratio_60, volume_z_20.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series, per_asset,
                         forward_returns, compute_ic, validate_factor, report,
                         VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------- active library (from artifacts) ----------------
ACTIVE = ["calmness_20", "dxy_beta_cond_60x20", "intraday_drift_20",
          "mom20_volproxy60", "usdjpy_beta_cond_120x60"]
lib = {}
for fid in ACTIVE:
    p = Path("factors") / f"{fid}.signal.npy"
    if p.exists():
        a = np.load(p)
        lib[fid] = pd.DataFrame(a, index=panel.index, columns=panel.columns)
print(f"active library artifacts: {list(lib.keys())}")


def stacked_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Spearman rho on the stacked (date x asset) signal values - matches gate behavior."""
    sa = a.stack()
    sb = b.stack()
    df = pd.concat([sa.rename("x"), sb.rename("y")], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    return float(df["x"].corr(df["y"], method="spearman"))


def library_corr_gate_style(cand: pd.DataFrame) -> dict:
    out = {}
    for fid, sig in lib.items():
        out[fid] = round(stacked_spearman(cand, sig), 4)
    mx = max((abs(v) for v in out.values()), default=0.0)
    return {"pairwise": out, "max_abs": round(mx, 4)}


# sanity check: reproduce the gate's gain_loss_20 vs mom20_volproxy60 rho (~0.604)
gl_arr = np.load("factors/gain_loss_20.signal.npy")
gl = pd.DataFrame(gl_arr, index=panel.index, columns=panel.columns)
print("[sanity] stacked rho gain_loss_20 vs mom20_volproxy60 =",
      round(stacked_spearman(gl, lib["mom20_volproxy60"]), 4), "(gate evicted at 0.604)")

# ---------------- data ----------------
def load_ohlc():
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        df = df.set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


OHLC = load_ohlc()


def build_ohlc_factor(func, name=""):
    out = {}
    for a in TRADABLES:
        df = OHLC[a].dropna()
        out[a] = func(df).reindex(panel.index)
    F = pd.DataFrame(out, index=panel.index)
    if name:
        vc = F.notna().sum()
        low = vc[vc < 200].index.tolist()
        print(f"  [cov:{name}] asset-days {F.notna().sum().sum()/(len(F)*15):.3f} "
              f"ge8 {F.notna().sum(axis=1).ge(8).mean():.3f} low-valid: {low}")
    return F


# ---------------- candidates ----------------
cands = {}

# A1. downside beta vs SPX (cycle 33)
spx_ret = panel["SPX"].dropna().pct_change()


def down_beta(s, w=60, minp=15):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if i < w - 1:
            continue
        seg = df.iloc[max(0, i - w + 1): i + 1]
        neg = seg[seg["m"] < 0]
        if len(neg) < minp:
            continue
        v = neg["m"].var()
        if v > 0:
            out.iloc[i] = neg["a"].cov(neg["m"]) / v
    return out.reindex(panel.index)


cands["downbeta_spx_60"] = per_asset(panel, down_beta)

# A2. lagged beta vs SPX (cycle 33)
def lag_beta(s, w=60, minp=30):
    ar = s.pct_change()
    spx_lag = spx_ret.shift(1)
    df = pd.concat([ar.rename("a"), spx_lag.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b.reindex(panel.index)


cands["lagbeta_spx_60"] = per_asset(panel, lag_beta)

# A3. volume-price feedback (cycle 33)
def vol_price_corr(df, w=60, minp=30):
    ar = df["close"].pct_change().abs()
    return ar.rolling(w, min_periods=minp).corr(df["volume"])


cands["vol_price_corr_60"] = build_ohlc_factor(vol_price_corr, "vol_price_corr_60")

# B1. overnight gap momentum (complement of intraday_drift)
def overnight_gap(df, w=20, mp=10):
    g = df["open"] / df["close"].shift(1) - 1.0
    return g.rolling(w, min_periods=mp).mean()


cands["overnight_gap_20"] = build_ohlc_factor(overnight_gap, "overnight_gap_20")

# B2. 20d high-low range position
def hilo_pos(df, w=20, mp=10):
    hi = df["high"].rolling(w, min_periods=mp).max()
    lo = df["low"].rolling(w, min_periods=mp).min()
    return (df["close"] - lo) / (hi - lo + 1e-12)


cands["hilo_pos_20"] = build_ohlc_factor(hilo_pos, "hilo_pos_20")

# B3. RSI-14
def rsi(df, w=14, mp=7):
    r = df["close"].pct_change()
    up = r.clip(lower=0).rolling(w, min_periods=mp).mean()
    dn = (-r.clip(upper=0)).rolling(w, min_periods=mp).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


cands["rsi_14"] = build_ohlc_factor(rsi, "rsi_14")

# B4. kurtosis 60d
def kurt(df, w=60, mp=40):
    return df["close"].pct_change().rolling(w, min_periods=mp).kurt()


cands["kurtosis_60"] = build_ohlc_factor(kurt, "kurtosis_60")

# B5. quantile tail ratio 60d (right vs left tail)
def tail_ratio(df, w=60, mp=40):
    r = df["close"].pct_change()

    def f(x):
        if np.isnan(x).any() or len(x) < mp:
            return np.nan
        q95, q50, q05 = np.nanpercentile(x, [95, 50, 5])
        return (q95 - q50) / (q50 - q05 + 1e-12)
    return r.rolling(w, min_periods=mp).apply(f, raw=True)


cands["tail_ratio_60"] = build_ohlc_factor(tail_ratio, "tail_ratio_60")

# B6. volume trend 20/60
def volume_z(df, ws=20, wl=60):
    v = df["volume"]
    return v.rolling(ws, min_periods=10).mean() / v.rolling(wl, min_periods=30).mean() - 1.0


cands["volume_z_20"] = build_ohlc_factor(volume_z, "volume_z_20")

# ---------------- validation ----------------
print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, F in cands.items():
    m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    # gate-style stacked rho replaces/augments per-date corr
    lc = library_corr_gate_style(F)
    m["max_abs_library_correlation"] = lc["max_abs"]
    m["library_pairwise_corr"] = lc["pairwise"]
    m["turnover_10d_rank"] = m.pop("turnover_10_rank", None)
    ic_ser = compute_ic(F, fwd_cache[str(ADM_H)]).dropna()
    reg = {}
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            reg[r0[:4] + ("-21" if r0[:4] == "2020" else "")] = {
                "ic": round(float(sub.mean()), 4),
                "icir": round(float(sub.mean() / sd) if sd > 0 else 0.0, 4),
                "n_dates": int(len(sub))}
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        reg["last250"] = {"ic": round(float(last.mean()), 4),
                          "icir": round(float(last.mean() / sd) if sd > 0 else 0.0, 4),
                          "n_dates": int(len(last))}
    p_ic = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    p_corr = abs(m["max_abs_library_correlation"]) < 0.5
    p = p_ic and p_corr
    print(f"[{name}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"cov_asset={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxrho={m['max_abs_library_correlation']} "
          f"=> {'PASS' if p else 'FAIL'}")
    print(f"    pairwise: {m['library_pairwise_corr']}")
    print(f"    regime: {json.dumps(reg)}")
    results[name] = {"metrics": m, "pass": bool(p), "regime": reg}

json.dump(results, open("scripts/_miner3_cycle34_results.json", "w"), indent=1, default=float)
print("\nDONE validation")
