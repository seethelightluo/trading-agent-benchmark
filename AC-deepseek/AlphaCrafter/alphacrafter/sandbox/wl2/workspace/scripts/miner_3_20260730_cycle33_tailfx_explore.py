"""miner_3 2026-07-30 cycle 33: orthogonal factor families vs 3-factor active library.

Active library: mom20_volproxy60, dxy_beta_cond_60x20, calmness_20.
Prior failures: downside_dev_60 (regime flip), intraday_cum_20 (rho 0.52 vs mom),
eff_ratio_20 (rho 0.524 vs mom), vol_surge_20 (rho 0.554 vs mom).

This cycle targets NEW axes:
  A. FX-carry conditional beta (USDJPY, EURUSD) - distinct macro axis vs DXY basket
  B. Tail co-movement with SPX (co-skewness, downside beta, asymmetric correlation)
  C. Vol term-structure slope (vol10/vol60, vol20/vol60) - regime-trend of vol
  D. Vol asymmetry (up semi-dev / down semi-dev) - crash-risk tilt
  E. Candle structure (body ratio, shadow ratios) - microstructure decisiveness
  F. Volume-price feedback (corr(volume, |ret|)) and lagged-market beta
"""
import sys, json
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

lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
print(f"library: {list(lib.keys())}; panel {panel.shape}; visible {VISIBLE_THROUGH}")


def load_ohlc():
    """Per-asset DataFrame with open/high/low/close/volume on own calendar."""
    out = {}
    for a in TRADABLES:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        df = df.set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


OHLC = load_ohlc()


def beta_to(s, m_ret, w=60, minp=30):
    """Rolling beta of asset daily returns on macro daily returns (own calendar)."""
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), m_ret.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b


def cond_beta_factor(macro_name, w=60, mw=20):
    """beta_w(asset, macro) x macro mw-day momentum."""
    m = macro_series(macro_name)
    m_ret = m.pct_change()
    m_mom = m / m.shift(mw) - 1.0
    parts = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        b = beta_to(s, m_ret, w)
        parts[a] = b.mul(m_mom.reindex(b.index), axis=0).reindex(panel.index)
    return pd.DataFrame(parts, index=panel.index)


def spx_cond_beta(w=60, mw=20):
    spx = panel["SPX"].dropna()
    spx_ret = spx.pct_change()
    spx_mom = spx / spx.shift(mw) - 1.0
    parts = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        b = beta_to(s, spx_ret, w)
        parts[a] = b.mul(spx_mom.reindex(b.index), axis=0).reindex(panel.index)
    return pd.DataFrame(parts, index=panel.index)


def co_skew_spx(s, spx_ret, w=60, minp=40):
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    mu_a = df["a"].rolling(w, min_periods=minp).mean()
    mu_m = df["m"].rolling(w, min_periods=minp).mean()
    cov3 = ((df["a"] - mu_a) * (df["m"] - mu_m) ** 2).rolling(w, min_periods=minp).mean()
    sig_a = df["a"].rolling(w, min_periods=minp).std()
    sig_m = df["m"].rolling(w, min_periods=minp).std()
    return (cov3 / (sig_a * sig_m ** 2 + 1e-12)).reindex(panel.index)


def down_beta_spx(s, spx_ret, w=60, minp=15):
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


def asym_corr_spx(s, spx_ret, w=60, minp=15):
    """corr(asset, SPX | SPX<0) - corr(asset, SPX | SPX>0): crisis correlation premium."""
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), spx_ret.rename("m")], axis=1).dropna()
    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if i < w - 1:
            continue
        seg = df.iloc[max(0, i - w + 1): i + 1]
        neg = seg[seg["m"] < 0]
        pos = seg[seg["m"] > 0]
        if len(neg) < minp or len(pos) < minp:
            continue
        cn = np.corrcoef(neg["a"], neg["m"])[0, 1] if neg["a"].std() > 0 else np.nan
        cp = np.corrcoef(pos["a"], pos["m"])[0, 1] if pos["a"].std() > 0 else np.nan
        if np.isfinite(cn) and np.isfinite(cp):
            out.iloc[i] = cn - cp
    return out.reindex(panel.index)


def lag_beta_spx(s, spx_ret, w=60, minp=30):
    """Rolling beta of asset ret on LAGGED (t-1) SPX ret: delayed reaction to market news."""
    ar = s.pct_change()
    spx_lag = spx_ret.shift(1)
    df = pd.concat([ar.rename("a"), spx_lag.rename("m")], axis=1).dropna()
    b = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return b.reindex(panel.index)


def vol_slope(s, w_short, w_long, minp=None):
    r = s.pct_change()
    vs = r.rolling(w_short, min_periods=max(10, w_short // 2)).std()
    vl = r.rolling(w_long, min_periods=max(20, w_long // 2)).std()
    return vs / vl


def updown_vol_ratio(s, w=60, minp=40):
    r = s.pct_change()
    up = r.where(r > 0, 0.0).pow(2).rolling(w, min_periods=minp).mean().pow(0.5)
    dn = r.where(r < 0, 0.0).pow(2).rolling(w, min_periods=minp).mean().pow(0.5)
    return up / (dn + 1e-12)


def candle_structure(mode, w=20, minp=12):
    """Per-asset candle stats from OHLC on own calendar."""
    out = {}
    for a in TRADABLES:
        df = OHLC[a].dropna()
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        if mode == "body":
            v = ((df["close"] - df["open"]).abs() / rng).rolling(w, min_periods=minp).mean()
        elif mode == "lower_shadow":
            v = ((df[["open", "close"]].min(axis=1) - df["low"]) / rng).rolling(w, min_periods=minp).mean()
        elif mode == "upper_shadow":
            v = ((df["high"] - df[["open", "close"]].max(axis=1)) / rng).rolling(w, min_periods=minp).mean()
        else:
            raise ValueError(mode)
        out[a] = v.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def vol_price_corr(s, vol_s, w=60, minp=30):
    """Rolling corr(volume, |daily return|): volume-volatility feedback strength."""
    ar = s.pct_change().abs()
    df = pd.concat([ar.rename("a"), vol_s.rename("v")], axis=1).dropna()
    c = df["a"].rolling(w, min_periods=minp).corr(df["v"])
    return c.reindex(panel.index)


def crash_speed_60(s, w=60, minp=40):
    """60d max drawdown depth divided by 60d realized vol: how FAST the drawdown was."""
    r = s.pct_change()
    vol = r.rolling(w, min_periods=minp).std()
    mdd = pd.Series(np.nan, index=r.index)
    xa = r.values
    for i in range(len(xa)):
        if i < minp - 1:
            continue
        seg = xa[max(0, i - w + 1): i + 1]
        if np.isnan(seg).any():
            continue
        peak = np.maximum.accumulate(seg)[-1]
        mdd.iloc[i] = (seg[-1] / peak - 1.0) if peak > 0 else np.nan
    return (mdd.abs() / (vol + 1e-12))


# ---------------- build candidates ----------------
cands = {}

# A. FX-carry conditional beta
cands["usdjpy_beta_cond_60x20"] = cond_beta_factor("USDJPY")
cands["eurusd_beta_cond_60x20"] = cond_beta_factor("EURUSD")

# B. Tail co-movement with SPX
spx_ret = panel["SPX"].dropna().pct_change()
cands["cosiskew_spx_60"] = per_asset(panel, co_skew_spx, spx_ret)
cands["downbeta_spx_60"] = per_asset(panel, down_beta_spx, spx_ret)
cands["asymcorr_spx_60"] = per_asset(panel, asym_corr_spx, spx_ret)
cands["spx_beta_cond_60x20"] = spx_cond_beta()
cands["lagbeta_spx_60"] = per_asset(panel, lag_beta_spx, spx_ret)

# C. Vol term-structure slope
cands["vol_slope_10_60"] = per_asset(panel, vol_slope, 10, 60)
cands["vol_slope_20_60"] = per_asset(panel, vol_slope, 20, 60)

# D. Vol asymmetry
cands["updown_vol_ratio_60"] = per_asset(panel, updown_vol_ratio, 60)
cands["crash_speed_60"] = per_asset(panel, crash_speed_60, 60)

# E. Candle structure
cands["body_ratio_20"] = candle_structure("body")
cands["lower_shadow_20"] = candle_structure("lower_shadow")
cands["upper_shadow_20"] = candle_structure("upper_shadow")

# F. Volume feedback
vol_panel = pd.DataFrame({a: OHLC[a]["volume"] for a in TRADABLES}, index=panel.index)
cands["vol_price_corr_60"] = per_asset(panel, vol_price_corr, vol_panel)

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    p = report(name, m)
    print(f"    decay: {m['decay_ic_by_horizon']}")
    print(f"    pairwise: {m.get('library_pairwise_corr')}")
    results[name] = {"metrics": m, "pass": p}

print("\n=== REGIME BREAKDOWN (10d IC by sub-period) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd_cache[str(ADM_H)]).dropna()
    parts = []
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            parts.append(f"{r0[:4]}:ic={sub.mean():+.4f}/icir={(sub.mean()/sd if sd>0 else 0):+.3f}/n={len(sub)}")
    last = ic_ser.iloc[-250:]
    if len(last) >= 30:
        sd = last.std()
        parts.append(f"last250:ic={last.mean():+.4f}/icir={(last.mean()/sd if sd>0 else 0):+.3f}/n={len(last)}")
    print(f"  {name:26s} | " + " | ".join(parts))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner3_cycle33_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle33")
