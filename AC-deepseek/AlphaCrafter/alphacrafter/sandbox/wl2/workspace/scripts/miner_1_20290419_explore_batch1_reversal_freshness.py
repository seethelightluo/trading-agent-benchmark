"""miner_1 2029-04-19: exploration batch 1 - reversal / trend-freshness / mean-reversion family.
Motivation (memory flags): momentum whipsaw persists (SOX/ETH/WTI trims regretted, ETH re-add
caught top); consider mean-reversion/freshness blend. Tests 8 candidates on full window
2020-01-01..2029-04-18 (visible through 2029-04-18). Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, HORIZON, MIN_ASSETS,
    load_asset, to_grid, load_macro,
    safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats,
)


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=2600)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        d = pd.DataFrame({
            "close": close, "ret": ret, "fwd10": close.shift(-HORIZON) / close - 1.0,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "volume": df["volume"].astype(float),
        })
        out[s] = d
    return out


series = asset_series_full()
print("assets with data:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

factors = {}

# 1. rev_5d: 5-day reversal (short-horizon mean reversion)
for s, df in series.items():
    factors.setdefault("rev_5d", {})[s] = -(df["close"] / df["close"].shift(5) - 1.0)

# 2. rev_3d: 3-day reversal
for s, df in series.items():
    factors.setdefault("rev_3d", {})[s] = -(df["close"] / df["close"].shift(3) - 1.0)

# 3. trend_age_20: days since 20d-return sign last flipped (0..20). Older trend = higher.
for s, df in series.items():
    m20 = df["close"] / df["close"].shift(20) - 1.0
    sign = np.sign(m20.fillna(0))
    flip = (sign.diff() != 0).astype(int)
    age = flip.groupby(flip.cumsum()).cumcount()
    factors.setdefault("trend_age_20", {})[s] = age.clip(upper=20)

# 4. flip_count_60: number of 10d-return sign flips in trailing 60d (choppiness, negated)
for s, df in series.items():
    m10 = df["close"] / df["close"].shift(10) - 1.0
    sign = np.sign(m10.fillna(0))
    flip = (sign.diff() != 0).astype(int)
    factors.setdefault("flip_count_60", {})[s] = -flip.rolling(60, min_periods=20).sum()

# 5. autocorr_1d_60: 1-day lag autocorrelation of daily returns over 60d (trend persistence)
for s, df in series.items():
    r = df["ret"]
    ac = r.rolling(60, min_periods=30).apply(lambda x: pd.Series(x).autocorr(1) if len(x) >= 30 else np.nan, raw=False)
    factors.setdefault("autocorr_1d_60", {})[s] = ac

# 6. boll_pos_20: Bollinger position (close - sma20)/std20
for s, df in series.items():
    c = df["close"]
    mu = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    factors.setdefault("boll_pos_20", {})[s] = pd.Series(safe_div(c - mu, sd), index=c.index)

# 7. rsi_14: classic 14d RSI (mean-reversion flavor; high = overbought)
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-r.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = pd.Series(safe_div(up, dn), index=r.index)
    rsi = 100 - 100 / (1 + rs)
    factors.setdefault("rsi_14", {})[s] = rsi

# 8. vol_term_20_60: vol20/vol60 (vol term structure; >1 = vol rising)
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    factors.setdefault("vol_term_20_60", {})[s] = pd.Series(safe_div(v20, v60), index=df.index)

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC")
        continue
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    reg250 = s["regime"].get("last250", {})
    print(f"{fid:20s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open("scripts/miner_1_20290419_batch1_results.json", "w"), indent=1, default=str)
print("DONE batch1")
