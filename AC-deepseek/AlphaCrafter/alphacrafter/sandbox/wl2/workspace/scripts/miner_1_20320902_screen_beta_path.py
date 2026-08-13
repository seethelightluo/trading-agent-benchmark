"""miner_1 2032-09-02 screening of NOVEL factor candidates (data visible through 2032-09-01).

Motivation: active 5f ensemble (spx_corr60, max_consec_gain_20, downbeta_spx_60,
mom_180d_skip5, range_pos_252) is momentum/beta heavy and ~9 months stale (2031-12-25);
memory flags commodity/crypto add whipsaw. The library (36 artifacts) already covers
SPX/DXY/USDJPY/VIX/BTC-beta and vol/range/momentum families. This batch targets axes NOT
in the library:

Group 1 - cross-asset reference BETA exposures (rate / gold / china / commodity / spread):
 1. us10y_beta_60     : rolling beta(asset_ret, US10Y_ret, 60)  - US rate sensitivity
 2. gold_beta_60      : rolling beta(asset_ret, XAU_ret, 60)    - real-asset/gold beta
 3. china_beta_60     : rolling beta(asset_ret, 000300.SH_ret, 60) - China equity exposure
 4. copper_beta_60    : rolling beta(asset_ret, COPPER_ret, 60) - commodity-cycle beta
 5. cnus_spread_beta_60: rolling beta(asset_ret, d(CN10Y-US10Y), 60) - China-US rate spread beta
    (self-reference asset set NaN to avoid auto-correlation 1.0 artifacts)

Group 2 - return-PATH statistics (orthogonal to level momentum / vol level):
 6. autocorr1_60      : 1-day return autocorrelation over 60d (trend persistence)
 7. var_ratio_5_60    : var(5d ret)/(5*var(1d ret)) over 60d (variance ratio, trend vs MR)
 8. kurt_60           : excess kurtosis of daily rets over 60d (tail thickness)
 9. amihud_60         : mean(|ret|/volume) over 60d (illiquidity proxy; volume available)
10. skew_60           : skewness of daily rets over 60d (longer than library ret_skew_10)
11. mom_252_skip20    : 252d momentum skipping last 20d (long-horizon trend, skip recent noise)

Gates: |IC|>=0.0070 and |ICIR|>=0.0840 (Spearman vs fwd10, daily cross-section, >=8 valid).
Persistence additionally prefers max_abs_library_correlation < 0.5 (ensemble selection rule).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, cross_sectional_rank,
    spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
    turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS, GRID, N_GRID,
)

DAYS = 4200
dates = np.array(GRID)

series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 120:
        print("skip", s, flush=True)
        continue
    close = df["close"].astype(float)
    d = pd.DataFrame({
        "close": close, "ret": close.pct_change(),
        "high": df["high"].astype(float), "low": df["low"].astype(float),
        "volume": df["volume"].astype(float),
    })
    series[s] = d
print("assets with data:", len(series), sorted(series.keys()), flush=True)

fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))

ret_mat = to_grid({s: df["ret"] for s, df in series.items()})


def rolling_beta_on_calendar(asset_df, ref_ser, win=60, minp=30, self_sym=None, name=None):
    """beta of asset daily ret on reference daily ret (reindexed to asset calendar)."""
    ref = ref_ser.reindex(asset_df.index)
    x = asset_df["ret"]
    y = ref
    cov = x.rolling(win, min_periods=minp).cov(y)
    var = y.rolling(win, min_periods=minp).var()
    beta = cov / var
    return beta


# ---------- reference return series (on each asset's own calendar) ----------
ref_ret = {}
for ref in ["US10Y", "XAU", "000300.SH", "COPPER", "CN10Y"]:
    if ref in series:
        ref_ret[ref] = series[ref]["ret"]
# CN10Y-US10Y spread level (close diff) and its daily change
spread = None
if "CN10Y" in series and "US10Y" in series:
    spread = series["CN10Y"]["close"] - series["US10Y"]["close"]

candidates = {}

# ---------- Group 1: reference betas ----------
beta_specs = [
    ("us10y_beta_60", "US10Y", "US10Y"),
    ("gold_beta_60", "XAU", "XAU"),
    ("china_beta_60", "000300.SH", "000300.SH"),
    ("copper_beta_60", "COPPER", "COPPER"),
]
for fid, ref_key, ref_sym in beta_specs:
    panel = {}
    for s, df in series.items():
        b = rolling_beta_on_calendar(df, ref_ret[ref_key])
        if s == ref_sym:
            b = pd.Series(np.nan, index=df.index)  # self-reference -> NaN
        panel[s] = b
    candidates[fid] = to_grid(panel)

# spread beta: beta of asset ret vs daily change of CN10Y-US10Y spread
if spread is not None:
    panel = {}
    for s, df in series.items():
        dspread = spread.reindex(df.index).diff()
        x = df["ret"]
        cov = x.rolling(60, min_periods=30).cov(dspread)
        var = dspread.rolling(60, min_periods=30).var()
        panel[s] = cov / var
    candidates["cnus_spread_beta_60"] = to_grid(panel)

# ---------- Group 2: path statistics ----------
# 6 autocorr1_60
panel = {}
for s, df in series.items():
    r = df["ret"]
    panel[s] = r.rolling(60, min_periods=40).apply(lambda z: pd.Series(z).autocorr(lag=1), raw=False)
candidates["autocorr1_60"] = to_grid(panel)

# 7 var_ratio_5_60 : mean(var of 5d ret) / (5 * var of 1d ret) over trailing 60d
panel = {}
for s, df in series.items():
    r = df["ret"]
    r5 = df["close"] / df["close"].shift(5) - 1.0
    var1 = r.rolling(5, min_periods=4).var()
    var5 = r5.rolling(5, min_periods=4).var()
    vr = var5 / (5.0 * var1)
    panel[s] = vr.rolling(60, min_periods=40).mean()
candidates["var_ratio_5_60"] = to_grid(panel)

# 8 kurt_60 (excess kurtosis, Fisher)
panel = {}
for s, df in series.items():
    r = df["ret"]
    panel[s] = r.rolling(60, min_periods=40).kurt()
candidates["kurt_60"] = to_grid(panel)

# 9 amihud_60 : mean(|ret|/volume) * 1e9
panel = {}
for s, df in series.items():
    ilq = (df["ret"].abs() / df["volume"].replace(0, np.nan)) * 1e9
    panel[s] = ilq.rolling(60, min_periods=40).mean()
candidates["amihud_60"] = to_grid(panel)

# 10 skew_60
panel = {}
for s, df in series.items():
    r = df["ret"]
    panel[s] = r.rolling(60, min_periods=40).skew()
candidates["skew_60"] = to_grid(panel)

# 11 mom_252_skip20
panel = {}
for s, df in series.items():
    c = df["close"]
    panel[s] = c.shift(20) / c.shift(252) - 1.0
candidates["mom_252_skip20"] = to_grid(panel)

# ---------- validation ----------
results = {}
for fid, mat in candidates.items():
    ics = spearman_ic_matrix(mat, fwd10)
    summ = summarize(ics, dates, fid, HORIZON)
    if summ is None:
        print(fid, "NO VALID IC DATES", flush=True)
        continue
    cov, dates_ge8 = coverage_stats(mat)
    to_ = turnover_10d_rank(cross_sectional_rank(mat))
    decay = decay_curve(mat, fwd_by_h)
    lpc, lpc_name, lpc_max = library_pairwise_corr(mat)
    ok = (abs(summ["ic"]) >= 0.0070) and (abs(summ["icir"]) >= 0.0840)
    persist_ok = ok and (lpc_max < 0.5)
    results[fid] = {
        "label": fid, "horizon": HORIZON,
        "n_ic_dates": summ["n_ic_dates"], "ic": round(summ["ic"], 4),
        "icir": round(summ["icir"], 4), "hit": round(summ["hit"], 3),
        "regime": summ["regime"],
        "max_abs_library_correlation": round(lpc_max, 4),
        "max_corr_with": lpc_name,
        "turnover_10d_rank": round(to_, 4),
        "coverage": round(cov, 4), "dates_ge8_frac": round(dates_ge8, 4),
        "decay": {k: round(v, 4) for k, v in decay.items()},
        "ok": ok, "persist_ok": persist_ok,
    }
    print("=" * 80, flush=True)
    print(fid, "| ic=%.4f icir=%.4f hit=%.3f n=%d | ok=%s persist_ok=%s" % (
        summ["ic"], summ["icir"], summ["hit"], summ["n_ic_dates"], ok, persist_ok), flush=True)
    print("   max_abs_library_correlation=%.3f (%s) | turnover=%.4f coverage=%.3f dates_ge8=%.3f"
          % (lpc_max, lpc_name, to_, cov, dates_ge8), flush=True)
    print("   decay:", {k: round(v, 4) for k, v in decay.items()}, flush=True)
    print("   regime:", json.dumps(summ["regime"]), flush=True)

with open("scripts/miner_1_20320902_screen_beta_path.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. summary of gate results:", flush=True)
for fid, r in results.items():
    print("  %-22s ic=%+.4f icir=%+.4f ok=%s persist_ok=%s maxcorr=%.3f" % (
        fid, r["ic"], r["icir"], r["ok"], r["persist_ok"], r["max_abs_library_correlation"]), flush=True)
