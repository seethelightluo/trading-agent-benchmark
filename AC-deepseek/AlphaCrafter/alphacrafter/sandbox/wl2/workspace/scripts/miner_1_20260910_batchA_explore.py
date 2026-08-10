"""miner_1 2026-09-10: batch exploration of novel factor families.
Candidate families NOT already in library:
 A. volatility term-structure ratio (compression/expansion)
 B. return autocorrelation / trend persistence
 C. mean-reversion z-score (vs momentum)
 D. RSI oscillator
 E. drawdown depth & recovery
 F. short-range high-low position
 G. return asymmetry (up/down ratio, win rate, skew60)
 H. cross-sectional relative momentum
 I. overnight drift
 J. coskew vs SPX (crash sensitivity)
 K. EURUSD conditional beta
Validation: daily cross-sectional Spearman IC vs fwd-10d (own calendar), |IC|>=0.0070, |ICIR|>=0.0840.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, GRID, GIDX, N_GRID, HORIZON, MIN_ASSETS,
    load_asset, to_grid, cross_sectional_rank, spearman_ic_matrix,
    summarize, fwd_by_horizon_dict, decay_curve, turnover_10d_rank,
    library_pairwise_corr, coverage_stats,
)

series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is None or len(df) < 300:
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    fwd = close.shift(-HORIZON) / close - 1.0
    d = pd.DataFrame({"close": close, "ret": ret, "fwd10": fwd,
                      "open": df["open"].astype(float),
                      "high": df["high"].astype(float),
                      "low": df["low"].astype(float)})
    series[s] = d
print(f"loaded {len(series)} assets")

fwd_grid = to_grid({s: d["fwd10"] for s, d in series.items()})
dates = np.array(GRID)

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)

def roll_std(x, w, minp=8):
    return x.rolling(w, min_periods=minp).std()

def evaluate(name, mat):
    ic_list = spearman_ic_matrix(mat, fwd_grid)
    summ = summarize(ic_list, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    turn = turnover_10d_rank(cross_sectional_rank(mat))
    decay = decay_curve(mat, fwd_by_horizon_dict(series))
    corr_map, mx_name, mx_val = library_pairwise_corr(mat)
    res = {
        "ic": summ["ic"], "icir": summ["icir"], "hit": summ["hit"],
        "n_ic_dates": summ["n_ic_dates"], "regime": summ.get("regime", {}),
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(turn, 4), "decay": decay,
        "max_abs_library_correlation": mx_val,
        "max_lib_corr_name": mx_name,
        "pass_gate": abs(summ["ic"]) >= 0.007 and abs(summ["icir"]) >= 0.084,
    }
    flag = "PASS" if res["pass_gate"] else "fail"
    print(f"[{flag}] {name}: ic={summ['ic']:.4f} icir={summ['icir']:.4f} "
          f"hit={summ['hit']:.3f} n={summ['n_ic_dates']} cov={cov_ad:.2f}/{cov_d8:.2f} "
          f"turn={turn:.3f} maxcorr={mx_val:.3f}({mx_name})")
    return res

results = {}

# ---------- A. vol term-structure ratios ----------
for s, d in series.items():
    v10 = roll_std(d["ret"], 10)
    v60 = roll_std(d["ret"], 60)
    d["vol_ts_10_60"] = safe_div(v10, v60)
    v5 = roll_std(d["ret"], 5)
    v40 = roll_std(d["ret"], 40)
    d["vol_ts_5_40"] = safe_div(v5, v40)
results["vol_ts_10_60"] = evaluate("vol_ts_10_60", to_grid({s: d["vol_ts_10_60"] for s, d in series.items()}))
results["vol_ts_5_40"] = evaluate("vol_ts_5_40", to_grid({s: d["vol_ts_5_40"] for s, d in series.items()}))

# ---------- B. autocorrelation / trend persistence ----------
for s, d in series.items():
    r = d["ret"]
    d["autocorr_1_20"] = r.rolling(20, min_periods=12).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 12 else np.nan, raw=False)
    mu = r.rolling(20, min_periods=10).mean()
    sd = roll_std(r, 20, 10)
    d["trend_ir_20"] = safe_div(mu, sd)
results["autocorr_1_20"] = evaluate("autocorr_1_20", to_grid({s: d["autocorr_1_20"] for s, d in series.items()}))
results["trend_ir_20"] = evaluate("trend_ir_20", to_grid({s: d["trend_ir_20"] for s, d in series.items()}))

# ---------- C. mean-reversion z-score ----------
for s, d in series.items():
    c = d["close"]
    for w in (60, 120):
        mu = c.rolling(w, min_periods=40).mean()
        sd = c.rolling(w, min_periods=40).std()
        d[f"zscore_{w}"] = safe_div(c - mu, sd)
results["zscore_60"] = evaluate("zscore_60", to_grid({s: d["zscore_60"] for s, d in series.items()}))
results["zscore_120"] = evaluate("zscore_120", to_grid({s: d["zscore_120"] for s, d in series.items()}))

# ---------- D. RSI 14 ----------
for s, d in series.items():
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=8).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=8).mean()
    rs = safe_div(up, dn)
    d["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
results["rsi_14"] = evaluate("rsi_14", to_grid({s: d["rsi_14"] for s, d in series.items()}))

# ---------- E. drawdown depth & recovery ----------
for s, d in series.items():
    c = d["close"]
    roll_max60 = c.rolling(60, min_periods=30).max()
    d["dd_depth_60"] = safe_div(c, roll_max60) - 1.0  # <=0
    roll_min120 = c.rolling(120, min_periods=60).min()
    d["recovery_120"] = safe_div(c, roll_min120) - 1.0  # >=0
results["dd_depth_60"] = evaluate("dd_depth_60", to_grid({s: d["dd_depth_60"] for s, d in series.items()}))
results["recovery_120"] = evaluate("recovery_120", to_grid({s: d["recovery_120"] for s, d in series.items()}))

# ---------- F. short-range HL position ----------
for s, d in series.items():
    hh = d["high"].rolling(20, min_periods=10).max()
    ll = d["low"].rolling(20, min_periods=10).min()
    d["hl_pos_20"] = safe_div(d["close"] - ll, hh - ll)
results["hl_pos_20"] = evaluate("hl_pos_20", to_grid({s: d["hl_pos_20"] for s, d in series.items()}))

# ---------- G. return asymmetry ----------
for s, d in series.items():
    r = d["ret"]
    up = r.clip(lower=0)
    dn = (-r).clip(lower=0)
    mup = up.rolling(20, min_periods=10).mean()
    mdn = dn.rolling(20, min_periods=10).mean()
    d["updown_ratio_20"] = safe_div(mup, mdn)
    d["win_rate_20"] = (r > 0).astype(float).rolling(20, min_periods=10).mean()
    mu = r.rolling(60, min_periods=40).mean()
    sd = roll_std(r, 60, 40)
    m3 = (r ** 3).rolling(60, min_periods=40).mean()
    d["skew_60"] = safe_div(m3 - 3 * mu * roll_std(r, 60, 40) ** 2 - mu ** 3, sd ** 3)
results["updown_ratio_20"] = evaluate("updown_ratio_20", to_grid({s: d["updown_ratio_20"] for s, d in series.items()}))
results["win_rate_20"] = evaluate("win_rate_20", to_grid({s: d["win_rate_20"] for s, d in series.items()}))
results["skew_60"] = evaluate("skew_60", to_grid({s: d["skew_60"] for s, d in series.items()}))

# ---------- H. cross-sectional relative momentum ----------
for w in (20, 60):
    mat = to_grid({s: d["close"].pct_change(w).rename(w) for s, d in series.items()})
    rel = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        ok = ~np.isnan(row)
        if ok.sum() >= MIN_ASSETS:
            rel[t, ok] = row[ok] - np.nanmedian(row[ok])
    results[f"rel_mom_{w}"] = evaluate(f"rel_mom_{w}", rel)

# ---------- I. overnight drift ----------
for s, d in series.items():
    prev_close = d["close"].shift(1)
    gap = safe_div(d["open"] - prev_close, prev_close)
    d["overnight_20"] = gap.rolling(20, min_periods=10).mean()
results["overnight_20"] = evaluate("overnight_20", to_grid({s: d["overnight_20"] for s, d in series.items()}))

# ---------- J. coskew vs SPX (crash sensitivity) ----------
spx_ret = series["SPX"]["ret"]
spx_sq = (spx_ret - spx_ret.mean()) ** 2
for s, d in series.items():
    r = d["ret"]
    cov = r.rolling(60, min_periods=40).cov(spx_sq)
    var = spx_sq.rolling(60, min_periods=40).var()
    d["coskew_spx_60"] = safe_div(cov, var)
results["coskew_spx_60"] = evaluate("coskew_spx_60", to_grid({s: d["coskew_spx_60"] for s, d in series.items()}))

# ---------- K. EURUSD conditional beta ----------
def load_macro_series(name):
    p = f"../persistent/index_data/{name}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return df["close"].astype(float)

eur = load_macro_series("EURUSD")
eur_ret = eur.pct_change()
eur_trend = eur.pct_change(20)
cond = (eur_trend > 0).astype(float).reindex(GRID)  # EURUSD uptrend regime
for s, d in series.items():
    r = d["ret"].reindex(GRID)
    betas = []
    up = (cond == 1).values
    for t in range(60, len(GRID)):
        w = slice(t - 60, t)
        rw = r.values[w]
        ew = eur_ret.reindex(GRID).values[w]
        mask = up[w] & ~np.isnan(rw) & ~np.isnan(ew)
        if mask.sum() >= 30:
            b = np.polyfit(ew[mask], rw[mask], 1)[0]
            betas.append(b)
        else:
            betas.append(np.nan)
    d["eurusd_beta_cond_120x60"] = pd.Series([np.nan] * 60 + betas, index=GRID)
results["eurusd_beta_cond_120x60"] = evaluate("eurusd_beta_cond_120x60",
    to_grid({s: d["eurusd_beta_cond_120x60"] for s, d in series.items()}))

json.dump({k: v for k, v in results.items() if v is not None},
          open("scripts/miner_1_20260910_batchA_results.json", "w"), indent=1)
print("SAVED batchA results")
