"""miner_1 cycle 2026-08-13: statistical/trend-quality factor family screen.

Universe: 15 tradable cross-asset instruments. Data: ../persistent/stock_data + index_data,
truncated to visible_through from ../persistent/date.json (2026-08-12). No lookahead:
factor at t uses data <= t; forward return t+1..t+h on each asset's OWN calendar.

Candidates (interpretable, per-asset cross-sectional):
  trend_r2_60     : R^2 of 60d log-price linear trend (trend quality)
  autocorr_20     : lag-1 return persistence proxy  mean(r*r1)/mean(r^2) over 20d
  skew_60         : Fisher skewness of daily returns, 60d
  kurt_60         : excess kurtosis of daily returns, 60d
  bollinger_pos_20: (close - SMA20) / (2*std20)  mean-reversion position
  drawdown_60     : close/rolling_max(close,60) - 1
  vol_trend_20x60 : volume 5d/60d ratio - 1  (liquidity flow)
  range_20        : mean((high-low)/close, 20) intraday amplitude
  gap_level_20    : mean(open/prev_close - 1, 20) overnight gap persistence
  vol_zscore_60   : z-score of 20d realized vol vs its 60d history
  rv_ratio_10x60  : std10/std60 (vol term-structure proxy)
  hi_lo_pos_20    : (close - min20)/(max20 - min20)  close location in 20d range
  wti_beta_60     : rolling beta to WTI returns, 60d
  btc_beta_60     : rolling beta to BTC returns, 60d
  gk_cc_ratio_60  : Garman-Klass vol / close-to-close vol, 60d (noise/efficiency)
  amihud_20       : mean(|ret|/volume, 20) illiquidity
  vol_slope_20x60 : (std20 - std60)/std60
  win_rate_20     : fraction of positive days over 20d
  close_loc_20    : mean((2C-H-L)/(H-L), 20) intraday close location

Gates (benchmark-wide): |IC| >= 0.0070 AND |ICIR| >= 0.0840 at h=10.
Library correlation vs existing effective factor .npy artifacts.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATE_STATE = json.load(open("../persistent/date.json"))
TRADING_DAYS = DATE_STATE["trading_days"]
VISIBLE = DATE_STATE["visible_through"]
ART_START = "2020-01-01"
ROW0 = TRADING_DAYS.index(ART_START)
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}")

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")
HORIZON = 10
MIN_ASSETS = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_asset(sym, cols=("open", "high", "low", "close", "volume")):
    p = (INDEX_DIR if sym in MACRO else DATA_DIR) / f"{sym}.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return pd.DataFrame(out, index=df.index)


def reindex_per_asset(panel_dict):
    return pd.DataFrame({a: s.reindex(GRID) for a, s in panel_dict.items()}, index=GRID)


def safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-12, np.nan, b)


# ---------------- load data ----------------
CLOSE = reindex_per_asset({a: load_asset(a)["close"] for a in TRADABLES})
OHLCV = {a: load_asset(a) for a in TRADABLES}
print("panel shape:", CLOSE.shape, "nan frac: %.3f" % CLOSE.isna().mean().mean())


def per_asset_panel(func, *args, **kwargs):
    """Apply func(series-of-one-asset) on own calendar, reindex to GRID."""
    out = {}
    for a in TRADABLES:
        df = OHLCV[a]
        s = func(df, *args, **kwargs)
        out[a] = s.reindex(GRID)
    return pd.DataFrame(out, index=GRID)


def own_ret(a):
    return OHLCV[a]["close"].pct_change()


def rolling_beta(asset_ret, mkt_ret, window, min_obs):
    df = pd.concat([asset_ret.rename("a"), mkt_ret.reindex(asset_ret.index).rename("m")], axis=1).dropna()
    cov = df["a"].rolling(window, min_periods=min_obs).cov(df["m"])
    var = df["m"].rolling(window, min_periods=min_obs).var()
    return safe_div(cov, var).reindex(asset_ret.index)


def close_ret_panel():
    return CLOSE.pct_change()


# ---------------- candidates ----------------
CAND = {}

# 1. trend_r2_60 : R^2 of log-price OLS trend over 60d (per-asset covariance, own calendar)
y = np.log(CLOSE)
cand_r2 = {}
for a in TRADABLES:
    ys = y[a].dropna()
    x = pd.Series(np.arange(len(ys)), index=ys.index)
    cxy = ys.rolling(60, min_periods=30).cov(x)
    vx = x.rolling(60, min_periods=30).var()
    vy = ys.rolling(60, min_periods=30).var()
    b = safe_div(cxy, vx)
    cand_r2[a] = safe_div(b ** 2 * vx, vy).reindex(GRID)
CAND["trend_r2_60"] = pd.DataFrame(cand_r2, index=GRID)

# 2. autocorr_20 : persistence proxy
ret = close_ret_panel()
CAND["autocorr_20"] = safe_div((ret * ret.shift(1)).rolling(20, min_periods=10).mean(),
                               (ret ** 2).rolling(20, min_periods=10).mean())

# 3/4. skew / kurt 60
CAND["skew_60"] = ret.rolling(60, min_periods=30).skew()
CAND["kurt_60"] = ret.rolling(60, min_periods=30).kurt()

# 5. bollinger_pos_20
ma20 = CLOSE.rolling(20, min_periods=10).mean()
sd20 = CLOSE.rolling(20, min_periods=10).std()
CAND["bollinger_pos_20"] = safe_div(CLOSE - ma20, 2 * sd20)

# 6. drawdown_60
CAND["drawdown_60"] = CLOSE / CLOSE.rolling(60, min_periods=30).max() - 1.0

# 7. vol_trend_20x60 (volume 5/60)
v5 = per_asset_panel(lambda df: df["volume"].rolling(5, min_periods=3).mean())
v60 = per_asset_panel(lambda df: df["volume"].rolling(60, min_periods=20).mean())
CAND["vol_trend_20x60"] = safe_div(v5, v60) - 1.0

# 8. range_20
rng = per_asset_panel(lambda df: safe_div(df["high"] - df["low"], df["close"]))
CAND["range_20"] = rng.rolling(20, min_periods=10).mean()

# 9. gap_level_20
gap = per_asset_panel(lambda df: safe_div(df["open"], df["close"].shift(1)) - 1.0)
CAND["gap_level_20"] = gap.rolling(20, min_periods=10).mean()

# 10. vol_zscore_60
rv20 = ret.rolling(20, min_periods=10).std()
rv20_m = rv20.rolling(60, min_periods=30).mean()
rv20_s = rv20.rolling(60, min_periods=30).std()
CAND["vol_zscore_60"] = safe_div(rv20 - rv20_m, rv20_s)

# 11. rv_ratio_10x60
CAND["rv_ratio_10x60"] = safe_div(ret.rolling(10, min_periods=8).std(),
                                  ret.rolling(60, min_periods=30).std())

# 12. hi_lo_pos_20
hi20 = CLOSE.rolling(20, min_periods=10).max()
lo20 = CLOSE.rolling(20, min_periods=10).min()
CAND["hi_lo_pos_20"] = safe_div(CLOSE - lo20, hi20 - lo20)

# 13/14. wti/btc beta 60
wti_ret = own_ret("WTI")
btc_ret = own_ret("BTC")
CAND["wti_beta_60"] = per_asset_panel(lambda df: rolling_beta(own_ret(df.index.name) if False else df["close"].pct_change(), wti_ret, 60, 30))
# rebuild wti_beta properly (per-asset loop)
wb = {}
bb = {}
for a in TRADABLES:
    ar = own_ret(a)
    wb[a] = rolling_beta(ar, wti_ret, 60, 30).reindex(GRID)
    bb[a] = rolling_beta(ar, btc_ret, 60, 30).reindex(GRID)
CAND["wti_beta_60"] = pd.DataFrame(wb, index=GRID)
CAND["btc_beta_60"] = pd.DataFrame(bb, index=GRID)

# 15. gk_cc_ratio_60 : Garman-Klass vs close-to-close vol
def gk_series(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    lp = np.log(c / o)
    hl = np.log(h / l)
    gk2 = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * lp ** 2
    gk2 = gk2.clip(lower=0)
    cc2 = np.log(c / c.shift(1)) ** 2
    gk_vol = np.sqrt(gk2.rolling(60, min_periods=30).mean())
    cc_vol = np.sqrt(cc2.rolling(60, min_periods=30).mean())
    return safe_div(gk_vol, cc_vol)
CAND["gk_cc_ratio_60"] = per_asset_panel(gk_series)

# 16. amihud_20
def amihud(df):
    v = df["volume"].replace(0, np.nan)
    return safe_div(df["close"].pct_change().abs(), v).rolling(20, min_periods=10).mean()
CAND["amihud_20"] = per_asset_panel(amihud)

# 17. vol_slope_20x60
CAND["vol_slope_20x60"] = safe_div(ret.rolling(20, min_periods=10).std() - ret.rolling(60, min_periods=30).std(),
                                   ret.rolling(60, min_periods=30).std())

# 18. win_rate_20
CAND["win_rate_20"] = (ret > 0).rolling(20, min_periods=10).mean()

# 19. close_loc_20
cl = per_asset_panel(lambda df: safe_div(2 * df["close"] - df["high"] - df["low"], df["high"] - df["low"]))
CAND["close_loc_20"] = cl.rolling(20, min_periods=10).mean()

print("candidates:", len(CAND))


# ---------------- forward returns & IC ----------------
def fwd_ret_panel(h):
    out = {}
    for a in TRADABLES:
        s = CLOSE[a].dropna()
        out[a] = (s.shift(-h) / s - 1.0).reindex(GRID)
    return pd.DataFrame(out, index=GRID)


FWD = {str(h): fwd_ret_panel(h) for h in (1, 2, 3, 5, 10, 20)}


def compute_ic(factor_panel, ret_panel):
    dates = factor_panel.index.intersection(ret_panel.index)
    Fr = factor_panel.loc[dates].rank(axis=1).values
    Rr = ret_panel.loc[dates].rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= MIN_ASSETS
    ics = np.full(len(dates), np.nan)
    idx = np.where(valid)[0]
    for i in idx:
        f = Fr[i, m[i]] - Fr[i, m[i]].mean()
        r = Rr[i, m[i]] - Rr[i, m[i]].mean()
        den = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / den if den > 0 else np.nan
    return pd.Series(ics, index=dates)


def turnover_rank(fp, step=10):
    ranked = fp.rank(axis=1, pct=True)
    vals = []
    for i in range(step, len(ranked), step):
        a, b = ranked.iloc[i - step], ranked.iloc[i]
        m = a.notna() & b.notna()
        if m.sum() >= MIN_ASSETS:
            vals.append(float((b[m] - a[m]).abs().mean()))
    return float(np.mean(vals)) if vals else float("nan")


def cov_stats(fp):
    total = len(fp) * 15
    valid = int(fp.notna().sum().sum())
    ge8 = int((fp.notna().sum(axis=1) >= MIN_ASSETS).sum())
    return round(valid / total, 4), round(ge8 / len(fp), 4), int(len(fp)), ge8


# ---------------- library artifacts ----------------
ART_DIR = Path("factors")
LIB_ART = {}
for f in sorted(ART_DIR.glob("*.signal.npy")):
    try:
        a = np.load(f, allow_pickle=True)
        if a.shape[1] == 15:
            n = min(a.shape[0], len(GRID))
            dates = GRID[:n]
            LIB_ART[f.stem.replace(".signal", "")] = pd.DataFrame(a[:n], index=pd.Index(dates), columns=TRADABLES)
    except Exception:
        pass
print("library artifacts loaded:", len(LIB_ART))


def panel_rank_corr(a, b):
    dates = a.index.intersection(b.index)
    Ar = a.loc[dates].rank(axis=1).values
    Br = b.loc[dates].rank(axis=1).values
    m = (~np.isnan(Ar)) & (~np.isnan(Br))
    valid = m.sum(axis=1) >= MIN_ASSETS
    cs = []
    idx = np.where(valid)[0]
    for i in idx:
        x = Ar[i, m[i]] - Ar[i, m[i]].mean()
        y = Br[i, m[i]] - Br[i, m[i]].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            cs.append((x * y).sum() / den)
    return float(np.mean(cs)) if cs else 0.0


# ---------------- validate ----------------
results = {}
for name, fp in CAND.items():
    ic_ser = compute_ic(fp, FWD["10"]).dropna()
    ic = float(ic_ser.mean())
    icir = float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 and ic_ser.std() > 0 else 0.0
    hit = float((np.sign(ic_ser) == np.sign(ic)).mean()) if ic != 0 else 0.0
    cov_a, cov_d, n_tot, n_ge8 = cov_stats(fp)
    to = turnover_rank(fp, step=HORIZON)
    decay = {str(h): round(float(compute_ic(fp, FWD[str(h)]).mean()), 4) for h in (1, 2, 3, 5, 10, 20)}
    # last-250d and last-60d IC (recency/regime)
    ic_last250 = float(ic_ser.iloc[-250:].mean())
    ic_last60 = float(ic_ser.iloc[-60:].mean())
    # library correlation
    libc = {fid: panel_rank_corr(fp, sig) for fid, sig in LIB_ART.items()}
    maxabs = max((abs(v) for v in libc.values()), default=0.0)
    passed = abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR
    results[name] = {
        "ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
        "n_ic_dates": int(len(ic_ser)), "coverage_asset_days": cov_a, "coverage_dates_ge8": cov_d,
        "n_dates_total": n_tot, "n_dates_ge8": n_ge8, "turnover_10d_rank": round(to, 3),
        "decay_ic_by_horizon": decay, "ic_last250": round(ic_last250, 4), "ic_last60": round(ic_last60, 4),
        "max_abs_library_correlation": round(maxabs, 4),
        "library_pairwise_corr": {k: round(v, 4) for k, v in libc.items()},
        "pass": passed,
    }
    print(f"[{name:16s}] IC={results[name]['ic']:+.4f} ICIR={results[name]['icir']:+.4f} "
          f"hit={results[name]['ic_hit_ratio']:.3f} n={results[name]['n_ic_dates']} "
          f"cov_a={cov_a:.3f} cov_d={cov_d:.3f} to={to:.3f} maxlib={maxabs:.3f} "
          f"last250={ic_last250:+.4f} last60={ic_last60:+.4f} => {'PASS' if passed else 'fail'}")

json.dump({"visible": VISIBLE, "grid_rows": len(GRID), "results": results},
          open("scripts/miner_1_20260813_screen_results.json", "w"), indent=1)
print("\nsaved scripts/miner_1_20260813_screen_results.json")
