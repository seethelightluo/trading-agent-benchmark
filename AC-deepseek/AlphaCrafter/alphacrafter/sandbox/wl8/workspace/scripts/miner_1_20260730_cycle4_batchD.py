"""miner_1 cycle 4 batch D: novel low-correlation factor screen.
Focus: constructions unlikely to correlate with active library
(mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).

Signal data ends at previous completed trading day 2026-07-29.
Forward returns computed on close data through 2026-07-30 (needed for fwd IC).
"""
import json
import base64
import zlib
import io
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
SIGNAL_END = pd.Timestamp("2026-07-29")   # last completed trading day
DATA_END = pd.Timestamp("2026-07-30")     # full close data for fwd returns
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS_PER_DATE = 8

# ---------------------------------------------------------------- data load
def load_ohlcv(end_date=DATA_END):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
        vols[a] = df["volume"].astype(float)
        opens[a] = df["open"].astype(float)
        highs[a] = df["high"].astype(float)
        lows[a] = df["low"].astype(float)
    return (pd.DataFrame(closes), pd.DataFrame(vols), pd.DataFrame(opens),
            pd.DataFrame(highs), pd.DataFrame(lows))

close, vol, open_, high, low = load_ohlcv()

def load_macro(name):
    df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= DATA_END].set_index("date").sort_index()
    return df["close"].astype(float)

dxy = load_macro("DXY")

def dense_asset(a):
    idx = close[a].dropna().index
    return {k: close[a].reindex(idx) for k in ["close"]} | {
        "vol": vol[a].reindex(idx), "open": open_[a].reindex(idx),
        "high": high[a].reindex(idx), "low": low[a].reindex(idx)}

def union_panel(series_dict):
    out = {}
    for a in ASSETS:
        s = pd.Series(series_dict[a], index=close[a].dropna().index)
        out[a] = s.reindex(close.index)
    return pd.DataFrame(out)

# ---------------------------------------------------------------- factor fns
def f_beta_asset(asset_ret, macro_ret, win=60):
    """rolling beta of asset returns to macro returns, on common trading days."""
    common = asset_ret.index.intersection(macro_ret.index)
    r_a = asset_ret.reindex(common).dropna()
    r_m = macro_ret.reindex(common).dropna()
    idx = r_a.index.intersection(r_m.index)
    r_a, r_m = r_a.reindex(idx), r_m.reindex(idx)
    cov = r_a.rolling(win, min_periods=40).cov(r_m)
    var = r_m.rolling(win, min_periods=40).var()
    return cov / var

def make_beta_factor(macro_close, win=60):
    out = {}
    for a in ASSETS:
        da = dense_asset(a)
        asset_ret = da["close"].pct_change()
        m_aligned = macro_close.reindex(da["close"].index).ffill()
        macro_ret = m_aligned.pct_change()
        out[a] = f_beta_asset(asset_ret, macro_ret, win)
    return union_panel(out)

def f_range_pos(close_, vol_, open_, high_, low_, win=60):
    hh = high_.rolling(win).max()
    ll = low_.rolling(win).min()
    return (close_ - ll) / (hh - ll)

def f_max_dd(close_, vol_, open_, high_, low_, win=60):
    roll_max = close_.rolling(win).max()
    return close_ / roll_max - 1.0   # negative drawdown depth

def f_vol_zscore(close_, vol_, vol__, open_, high_, low_, short=20, long=60):
    v = vol_
    return v.rolling(short).mean() / v.rolling(long).mean() - 1.0

def f_trend_r2(close_, vol_, open_, high_, low_, win=60):
    logp = np.log(close_)
    x = np.arange(win)
    x = x - x.mean()
    def r2(s):
        y = s.values
        if len(y) < win or not np.all(np.isfinite(y)):
            return np.nan
        b = np.polyfit(x, y, 1)
        pred = b[0] * x + b[1]
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return logp.rolling(win).apply(r2, raw=False)

def f_gk_vol_ratio(close_, vol_, open_, high_, low_, short=10, long=60):
    # Garman-Klass vol (per-day), then ratio of rolling means
    log_h = np.log(high_)
    log_l = np.log(low_)
    log_c = np.log(close_)
    log_o = np.log(open_)
    gk = 0.5 * (log_h - log_l) ** 2 - (2 * np.log(2) - 1) * (log_c - log_o) ** 2
    gk = np.sqrt(np.clip(gk, 0, None))
    return gk.rolling(short).mean() / gk.rolling(long).mean()

def f_body_ratio(close_, vol_, open_, high_, low_, win=20):
    body = (close_ - open_).abs()
    rng = (high_ - low_).replace(0, np.nan)
    return (body / rng).rolling(win).mean()

def f_shadow_asym(close_, vol_, open_, high_, low_, win=20):
    up_sh = high_ - np.maximum(open_, close_)
    lo_sh = np.minimum(open_, close_) - low_
    rng = (high_ - low_).replace(0, np.nan)
    return ((up_sh - lo_sh) / rng).rolling(win).mean()

def f_overnight_mom(close_, vol_, open_, high_, low_, win=20):
    gap = open_ / close_.shift(1) - 1.0
    return gap.rolling(win).mean()

# ---------------------------------------------------------------- validation
def fwd_returns(close_df, horizon):
    out = {}
    for a in ASSETS:
        c = close[a].dropna()
        fr = (c.shift(-horizon) / c - 1.0).reindex(close_df.index)
        out[a] = fr
    return pd.DataFrame(out)

def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x, y = factor.loc[dt], fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            ics.append(x[m].rank().corr(y[m].rank()))
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def turnover_rank(factor, lag=10):
    ranks = factor.rank(axis=1)
    return float(ranks.diff(lag).abs().mean(axis=1).dropna().mean())

def coverage(factor):
    n_total = float(factor.notna().sum().sum())
    denom = factor.shape[0] * factor.shape[1]
    ge8 = float((factor.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    return n_total / denom, ge8

def load_active_library():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        try:
            d = json.load(open(f"factors/{fid}.json"))
            raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
            p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                            index_col=0, parse_dates=True)
            lib[fid] = p
        except Exception as e:
            print(f"  [warn] lib {fid}: {e}")
    return lib

LIB = load_active_library()

def max_library_corr(panel):
    best = 0.0
    for fid, lp in LIB.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        if len(common) < 60 or len(cols) < 5:
            continue
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            continue
        rho = float(np.corrcoef(a[m], b[m])[0, 1])
        best = max(best, abs(rho))
    return best

def validate(panel, horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10):
    # only use signals through last completed day
    panel = panel[panel.index <= SIGNAL_END]
    cov_ad, cov_ge8 = coverage(panel)
    decay, ic_by_h = {}, {}
    for h in horizons:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr)
        ic_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic_main = ic_by_h[admission_horizon]
    ic = float(ic_main.mean())
    icir = float(ic_main.mean() / ic_main.std()) if len(ic_main) > 2 else np.nan
    hit = float((ic_main > 0).mean()) if np.isfinite(ic) else np.nan
    if ic < 0:
        hit = float((ic_main < 0).mean())
    res = {
        "ic": ic, "icir": icir, "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ic_main)),
        "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov_ge8, 4),
        "turnover_10d_rank": round(turnover_rank(panel), 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in horizons},
        "max_abs_library_correlation": round(max_library_corr(panel), 4),
    }
    return res

# ---------------------------------------------------------------- run
CANDIDATES = {}

# 1-5: cross-asset betas (each asset's rolling beta to a specific macro series)
CANDIDATES["wti_beta_60"] = make_beta_factor(close["WTI"], 60)
CANDIDATES["xau_beta_60"] = make_beta_factor(close["XAU"], 60)
CANDIDATES["btc_beta_60"] = make_beta_factor(close["BTC"], 60)
CANDIDATES["cn_beta_60"] = make_beta_factor(close["HSI"], 60)
CANDIDATES["usd_beta_60"] = make_beta_factor(dxy, 60)

# 6-13: structure/volume/trend-quality factors on dense per-asset calendar
def run_dense(fn, **kw):
    out = {}
    for a in ASSETS:
        da = dense_asset(a)
        try:
            s = fn(da["close"], da["vol"], da["open"], da["high"], da["low"], **kw)
            out[a] = s
        except Exception:
            out[a] = pd.Series(np.nan, index=da["close"].index)
    return union_panel(out)

CANDIDATES["range_pos_60"] = run_dense(f_range_pos, win=60)
CANDIDATES["max_dd_60"] = run_dense(f_max_dd, win=60)
CANDIDATES["vol_zscore_20x60"] = run_dense(f_vol_zscore)
CANDIDATES["trend_r2_60"] = run_dense(f_trend_r2, win=60)
CANDIDATES["gk_vol_ratio_10x60"] = run_dense(f_gk_vol_ratio)
CANDIDATES["body_ratio_20"] = run_dense(f_body_ratio, win=20)
CANDIDATES["shadow_asym_20"] = run_dense(f_shadow_asym, win=20)
CANDIDATES["overnight_mom_20"] = run_dense(f_overnight_mom, win=20)

print(f"Union panel dates: {close.shape[0]}, assets: {close.shape[1]}")
print(f"Signal dates through {SIGNAL_END.date()}")

results = {}
for name, panel in CANDIDATES.items():
    res = validate(panel)
    results[name] = res
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"\n=== {name} ===")
    for k in ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "coverage_asset_days",
              "coverage_dates_ge8", "turnover_10d_rank", "max_abs_library_correlation"]:
        print(f"  {k}: {res[k]}")
    print(f"  decay: {res['decay_ic_by_horizon']}")
    print(f"  GATE(|IC|>={IC_GATE},|ICIR|>={ICIR_GATE}): {'PASS' if ok else 'FAIL'}")

with open("scripts/_miner1_cycle4_batchD_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nSaved scripts/_miner1_cycle4_batchD_results.json")
