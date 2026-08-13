"""miner_1 2032-02-19 screening of NOVEL factor candidates (data visible through 2032-02-18).

Motivation: memory flags ensemble stale and commodity/crypto add whipsaw; library is dense with
momentum/vol/range/beta/corr concepts (35 signal artifacts). This batch targets orthogonal
concepts NOT directly covered, grouped by theme:

Group A - trend PATH quality (price-only, no simple momentum/range):
 1. whipsaw_10x60     : # sign flips of 10d returns over trailing 60d (choppiness; high=smooth trend)
 2. aroon_25          : 100*(25-dsh)/25 - 100*(25-dsl)/25 (high/low recency balance)
 3. adx_14            : Wilder ADX(14) trend strength (direction-agnostic)
 4. range_exp_20x60   : (max20-min20)/(max60-min60) range expansion ratio
 5. macd_hist_12_26_9 : (EMA12-EMA26) - EMA9(MACD), normalized by close
 6. eff_ratio_60      : |close_t-close_{t-60}| / sum|ret| over 60d (long-window efficiency)

Group B - cross-asset regime/risk (SPX / equal-weight references):
 7. connect_60        : mean absolute pairwise corr of asset vs all other 14 (systemic-ness)
 8. corr_delta_spx_60x60 : spx_corr60(t) - spx_corr60(t-60) (correlation regime CHANGE)
 9. ew_beta_60        : beta of asset vs equal-weight 15-asset portfolio (global risk exposure)
10. beta_gamma_60     : up_beta - down_beta vs SPX (leverage/asymmetry)
11. corr_asym_60      : corr(SPX-down days) - corr(SPX-up days) (tail comovement asymmetry)
12. lead_spx_60       : beta(SPX_t on asset_{t-1}) - does asset LEAD the market? (reverse of lagbeta)

Group C - vol regime/events:
13. vol_pct_252       : percentile rank of 20d realized vol within trailing 252d
14. vol_spike_age_60  : days since 20d vol > 1.5x 60d mean vol (vol-event recency, capped 60)
15. rev5_vol20        : -5d return / 20d realized vol (risk-adjusted short reversal)

Gates: |IC|>=0.0070, |ICIR|>=0.0840 (Spearman vs fwd10, daily cross-section, >=8 valid).
Persistence additionally requires max_abs_library_correlation < 0.5 (library conflict gate).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, cross_sectional_rank,
    spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
    turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS, GRID,
)

DAYS = 2100
dates = np.array(GRID)

series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 120:
        print("skip", s)
        continue
    close = df["close"].astype(float)
    d = pd.DataFrame({
        "close": close, "ret": close.pct_change(),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float),
    })
    series[s] = d
print("assets with data:", len(series), sorted(series.keys()), flush=True)

fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))

# aligned return matrix on master grid (for cross-asset factors)
ret_mat = to_grid({s: df["ret"] for s, df in series.items()})
spx_ret = ret_mat[:, ASSETS.index("SPX")]

candidates = {}

# ---------- Group A ----------
# 1 whipsaw_10x60
panel = {}
for s, df in series.items():
    r10 = df["close"] / df["close"].shift(10) - 1.0
    sg = np.sign(r10)
    flips = (sg.diff() != 0) & sg.notna() & sg.shift().notna()
    panel[s] = flips.rolling(60, min_periods=30).sum()
candidates["whipsaw_10x60"] = to_grid(panel)

# 2 aroon_25
def aroon(close, win=25):
    n = len(close)
    out = np.full(n, np.nan)
    vals = close.values.astype(float)
    for i in range(win, n):
        w = vals[i - win:i + 1]
        if not np.all(np.isfinite(w)):
            continue
        dsh = win - int(np.argmax(w[::-1]))   # days since window high
        dsl = win - int(np.argmin(w[::-1]))   # days since window low
        out[i] = 100.0 * (dsh - dsl) / win
    return pd.Series(out, index=close.index)

panel = {s: aroon(df["close"]) for s, df in series.items()}
candidates["aroon_25"] = to_grid(panel)

# 3 adx_14 (Wilder)
def adx(df, win=14):
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = tr.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean()
    pdi = 100.0 * plus_dm.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean() / atr.replace(0, np.nan)
    mdi = 100.0 * minus_dm.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out = dx.ewm(alpha=1.0 / win, adjust=False, min_periods=win).mean()
    return out

panel = {s: adx(df) for s, df in series.items()}
candidates["adx_14"] = to_grid(panel)

# 4 range_exp_20x60
panel = {}
for s, df in series.items():
    c = df["close"]
    rng20 = c.rolling(20, min_periods=10).max() - c.rolling(20, min_periods=10).min()
    rng60 = c.rolling(60, min_periods=30).max() - c.rolling(60, min_periods=30).min()
    panel[s] = rng20 / rng60.replace(0, np.nan)
candidates["range_exp_20x60"] = to_grid(panel)

# 5 macd_hist_12_26_9
panel = {}
for s, df in series.items():
    c = df["close"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    panel[s] = (macd - sig) / c.replace(0, np.nan)
candidates["macd_hist_12_26_9"] = to_grid(panel)

# 6 eff_ratio_60
panel = {}
for s, df in series.items():
    c = df["close"]
    net = (c - c.shift(60)).abs()
    path = df["ret"].abs().rolling(60, min_periods=30).sum()
    panel[s] = net / path.replace(0, np.nan)
candidates["eff_ratio_60"] = to_grid(panel)

# ---------- Group B ----------
# 7 connect_60
T, n = ret_mat.shape
conn = np.full_like(ret_mat, np.nan)
for t in range(59, T):
    w = ret_mat[t - 59:t + 1]
    ok_col = ~np.isnan(w).all(axis=0)
    if ok_col.sum() < MIN_ASSETS:
        continue
    ww = w[:, ok_col]
    m = ~np.isnan(ww).any(axis=1)
    if m.sum() < 30:
        continue
    cm = np.corrcoef(ww[m].T)
    if cm.shape[0] < 2:
        continue
    j = 0
    for k, ok in enumerate(ok_col):
        if ok:
            others = np.delete(cm[j], j)
            conn[t, k] = np.nanmean(np.abs(others))
            j += 1
candidates["connect_60"] = conn

# 8 corr_delta_spx_60x60
panel = {}
spx_r = series["SPX"]["ret"]
for s, df in series.items():
    corr = df["ret"].rolling(60, min_periods=15).corr(spx_r.reindex(df.index))
    panel[s] = corr - corr.shift(60)
candidates["corr_delta_spx_60x60"] = to_grid(panel)

# 9 ew_beta_60
ew = np.nanmean(np.where(np.isnan(ret_mat), np.nan, ret_mat), axis=1)
ew_valid = (np.sum(~np.isnan(ret_mat), axis=1) >= MIN_ASSETS)
ew[~ew_valid] = np.nan
ew_beta = np.full_like(ret_mat, np.nan)
for t in range(59, T):
    w = ret_mat[t - 59:t + 1]
    ewv = ew[t - 59:t + 1]
    for j in range(n):
        x = w[:, j]; y = ewv
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30 or np.std(y[m]) < 1e-12:
            continue
        ew_beta[t, j] = np.cov(x[m], y[m])[0, 1] / np.var(y[m])
candidates["ew_beta_60"] = ew_beta

# 10 beta_gamma_60 (up beta - down beta vs SPX)
def roll_beta_split(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        up = (x > 0) & ~np.isnan(x) & ~np.isnan(y)
        dn = (x < 0) & ~np.isnan(x) & ~np.isnan(y)
        if up.sum() >= minp and np.std(x[up]) > 1e-12:
            bu = np.cov(x[up], y[up])[0, 1] / np.var(x[up])
        else:
            bu = np.nan
        if dn.sum() >= minp and np.std(x[dn]) > 1e-12:
            bd = np.cov(x[dn], y[dn])[0, 1] / np.var(x[dn])
        else:
            bd = np.nan
        if np.isfinite(bu) and np.isfinite(bd):
            out.iloc[i] = bu - bd
    return out

panel = {}
for s, df in series.items():
    panel[s] = roll_beta_split(df["ret"], spx_r, 60, 12)
candidates["beta_gamma_60"] = to_grid(panel)

# 11 corr_asym_60
def roll_corr_split(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        up = (x > 0) & ~np.isnan(x) & ~np.isnan(y)
        dn = (x < 0) & ~np.isnan(x) & ~np.isnan(y)
        cu = np.corrcoef(x[up], y[up])[0, 1] if up.sum() >= minp else np.nan
        cd = np.corrcoef(x[dn], y[dn])[0, 1] if dn.sum() >= minp else np.nan
        if np.isfinite(cu) and np.isfinite(cd):
            out.iloc[i] = cd - cu
    return out

panel = {}
for s, df in series.items():
    panel[s] = roll_corr_split(df["ret"], spx_r, 60, 10)
candidates["corr_asym_60"] = to_grid(panel)

# 12 lead_spx_60 (asset leads SPX: beta of SPX_t on asset_{t-1})
lead = np.full_like(ret_mat, np.nan)
for j in range(n):
    a_lag = np.roll(ret_mat[:, j], 1)
    a_lag[0] = np.nan
    for t in range(59, T):
        x = a_lag[t - 59:t + 1]
        y = spx_ret[t - 59:t + 1]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30 or np.std(x[m]) < 1e-12:
            continue
        lead[t, j] = np.cov(x[m], y[m])[0, 1] / np.var(x[m])
candidates["lead_spx_60"] = lead

# ---------- Group C ----------
# 13 vol_pct_252
panel = {}
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=15).std()
    out = v20.rolling(252, min_periods=120).apply(
        lambda w: float(np.mean(w <= w[-1])), raw=True)
    panel[s] = out
candidates["vol_pct_252"] = to_grid(panel)

# 14 vol_spike_age_60
panel = {}
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=15).std()
    m60 = v20.rolling(60, min_periods=30).mean()
    spike = (v20 > 1.5 * m60).fillna(False).astype(int)
    age = np.full(len(spike), 60.0)
    last = -1
    for i in range(len(spike)):
        if spike.iloc[i] == 1:
            last = i
        age[i] = 60.0 if last < 0 else float(i - last)
    panel[s] = pd.Series(np.minimum(age, 60.0), index=df.index)
candidates["vol_spike_age_60"] = to_grid(panel)

# 15 rev5_vol20
panel = {}
for s, df in series.items():
    r5 = df["close"] / df["close"].shift(5) - 1.0
    v20 = df["ret"].rolling(20, min_periods=15).std()
    panel[s] = -r5 / v20.replace(0, np.nan)
candidates["rev5_vol20"] = to_grid(panel)

# ---------- evaluate ----------
results = {}
for fid, mat in candidates.items():
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC DATES", flush=True)
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
    s["persist_ok"] = bool(s["ok"] and max_rho < 0.5)
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    reg = s["regime"].get("last250", {})
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"({rho_name}) l250={reg.get('ic','NA')} ok={s['ok']} persist_ok={s['persist_ok']}", flush=True)

json.dump(results, open("scripts/miner_1_20320219_screen_novel.json", "w"), indent=1, default=str)
print("DONE saved scripts/miner_1_20320219_screen_novel.json")
