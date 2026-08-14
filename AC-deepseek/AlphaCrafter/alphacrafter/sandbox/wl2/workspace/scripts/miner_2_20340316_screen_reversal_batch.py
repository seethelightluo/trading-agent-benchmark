"""miner_2 2034-03-16: screen reversal / mean-reversion factor family (no persistence yet).
Full sample 2020-01-01..2034-03-15 (VISIBLE). 15-asset tradable cross-asset universe.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 on 10d forward returns (daily cross-sectional Spearman).
Rationale: ensemble repeatedly whipsawed by momentum adds (memory cycles 97/101/104/105/120);
mean-reversion overlays are the flagged hedge. Macro truncated at VISIBLE.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner_3_20260813_lib import (
    ASSETS, GRID, N_GRID, VISIBLE, load_asset, load_macro, safe_div,
    cross_sectional_rank, spearman_ic_matrix, summarize, decay_curve,
    fwd_by_horizon_dict, turnover_10d_rank, library_pairwise_corr,
    coverage_stats, HORIZON, MIN_ASSETS,
)

VIS = pd.Timestamp(VISIBLE)

series = {}
for s in ASSETS:
    df = load_asset(s, days=4000)
    if df is None or len(df) < 200:
        print("SKIP", s); continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    series[s] = pd.DataFrame({"close": close, "ret": ret})

print("assets loaded:", sorted(series.keys()))
print("n_assets:", len(series), "| grid rows:", N_GRID, "| visible:", VISIBLE)

fwd = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))

factors = {}

def add_factor(fid, panel):
    factors.setdefault(fid, {}).update(panel)

# 1 rev10_skip5: -(10d return skipping 5) short-term reversal
for s, df in series.items():
    r = df["close"] / df["close"].shift(11) - 1.0
    add_factor("rev10_skip5", {s: -r})

# 2 rev20_skip10: -(20d return skipping 10) medium reversal
for s, df in series.items():
    r = df["close"] / df["close"].shift(21) - 1.0
    add_factor("rev20_skip10", {s: -r})

# 3 bollinger_pos_20: (close - ma20) / (2*std20); negative = oversold (expect +IC after -1 sign)
for s, df in series.items():
    c = df["close"]
    ma = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    add_factor("bollinger_pos_20", {s: safe_div(c - ma, 2.0 * sd)})

# 4 rsi_14: standard RSI
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-r.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = safe_div(up, dn)
    add_factor("rsi_14", {s: 100.0 - 100.0 / (1.0 + rs)})

# 5 dd_depth_60: (close - max60)/max60 drawdown depth (negative = deeper drawdown)
for s, df in series.items():
    mx = df["close"].rolling(60, min_periods=30).max()
    add_factor("dd_depth_60", {s: safe_div(df["close"] - mx, mx)})

# 6 mom60_vol60: 60d momentum / 60d realized vol (vol-scaled trend)
for s, df in series.items():
    r60 = df["close"] / df["close"].shift(60) - 1.0
    v60 = df["ret"].rolling(60, min_periods=30).std()
    add_factor("mom60_vol60", {s: safe_div(r60, v60)})

# 7 vol_ratio_5x60: 5d vol / 60d vol (vol regime / short-term vol spike)
for s, df in series.items():
    v5 = df["ret"].rolling(5, min_periods=3).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    add_factor("vol_ratio_5x60", {s: safe_div(v5, v60)})

# 8 ret_skew_60: realized skewness of daily returns over 60d
for s, df in series.items():
    r = df["ret"]
    mu = r.rolling(60, min_periods=30).mean()
    sd = r.rolling(60, min_periods=30).std()
    mom3 = ((r - mu) ** 3).rolling(60, min_periods=30).mean()
    add_factor("ret_skew_60", {s: safe_div(mom3, sd ** 3)})

# 9 drawup_60: recovery position (close - min60)/(max60 - min60)
for s, df in series.items():
    hi = df["close"].rolling(60, min_periods=30).max()
    lo = df["close"].rolling(60, min_periods=30).min()
    add_factor("drawup_60", {s: safe_div(df["close"] - lo, hi - lo)})

# 10 dd_speed_60: depth normalized by days-since-high (fast deep drops = oversold)
for s, df in series.items():
    mx = df["close"].rolling(60, min_periods=30).max()
    depth = safe_div(df["close"] - mx, mx)  # <= 0
    is_high = (df["close"] >= mx).astype(float)
    days_since = is_high[::-1].cumsum()[::-1]  # count since last touch of running max
    days_since = days_since.replace(0, np.nan)
    add_factor("dd_speed_60", {s: safe_div(depth, days_since)})

# ---- validation ----
results = {}
for fid, panel in factors.items():
    mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for j, s in enumerate(ASSETS):
        if s in panel:
            mat[:, j] = panel[s].reindex(GRID).values
    rm = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    if not ics:
        print(fid, "NO IC DATES"); continue
    summ = summarize(ics, GRID, fid, HORIZON)
    if summ is None:
        print(fid, "SUMMARY NONE"); continue
    decay = decay_curve(mat, fwd)
    turn = turnover_10d_rank(rm)
    cov_asset_days, cov_dates_ge8 = coverage_stats(mat)
    corr_dict, max_name, max_abs = library_pairwise_corr(mat)
    results[fid] = {
        "label": fid, "horizon": HORIZON, "n_ic_dates": summ["n_ic_dates"],
        "ic": round(summ["ic"], 5), "icir": round(summ["icir"], 5),
        "hit": round(summ["hit"], 4),
        "regime": summ["regime"], "decay": decay,
        "turnover_10d_rank": round(turn, 4),
        "coverage": round(cov_asset_days, 4), "dates_ge8_frac": round(cov_dates_ge8, 4),
        "max_abs_library_correlation": round(max_abs, 4),
        "max_corr_with": max_name,
        "ok": (abs(summ["ic"]) >= 0.007 and abs(summ["icir"]) >= 0.084),
    }
    print("=" * 80)
    print(fid, "| IC", round(summ["ic"], 4), "| ICIR", round(summ["icir"], 4),
          "| hit", round(summ["hit"], 3), "| n", summ["n_ic_dates"],
          "| turn", round(turn, 3), "| cov", round(cov_asset_days, 3),
          "| maxlibcorr", round(max_abs, 3), "|", max_name)
    print("  regime:", {k: v for k, v in summ["regime"].items()})
    print("  decay:", decay)
    print("  GATE:", "PASS" if results[fid]["ok"] else "FAIL")

with open("scripts/miner_2_20340316_screen_reversal_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nSaved results json.")
