"""miner_1 2026-12-03: batch exploration of candidate factor families.
Visible data through previous completed trading day (2026-11-18)."""
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
INDEX_DATA_DIR = "../persistent/index_data/"
MIN_ASSETS = 8
H = 10  # admission horizon


def load_macro(name, max_date=None, days=1500):
    df = pd.read_csv(f"{INDEX_DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    if max_date is not None:
        df = df[df["date"] <= max_date]
    return df.tail(days).reset_index(drop=True)


def load_asset(symbol, days=1500):
    df = None
    try:
        df = get_index_daily_data(symbol=symbol, days=days)
    except Exception:
        df = None
    if df is None:
        try:
            df = get_stock_daily_data(symbol=symbol, days=days)
        except Exception:
            df = None
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_panel(days=1500):
    closes, vols, dates = {}, {}, None
    for s in WATCHLIST:
        df = load_asset(s, days=days)
        if df is None or len(df) < 200:
            print(f"  !! {s} insufficient data: {None if df is None else len(df)}")
            continue
        closes[s] = df.set_index("date")["close"]
        if "volume" in df.columns and df["volume"].notna().sum() > 0:
            vols[s] = df.set_index("date")["volume"]
        dts = df["date"].dt.normalize()
        if dates is None:
            dates = dts
    panel = pd.DataFrame(closes).sort_index()
    vpanel = pd.DataFrame(vols).sort_index()
    return panel, vpanel


def macropanel(days=1500, max_date=None):
    d = {}
    for m in MACRO:
        df = load_macro(m, max_date=max_date, days=days)
        d[m] = df.set_index("date")["close"]
    return pd.DataFrame(d).sort_index()


def fwd_ret(panel, h=H):
    return panel.shift(-h) / panel - 1.0


def spearman_ic(fs, fr, min_assets=MIN_ASSETS):
    ics, dates = [], []
    common = fs.index.intersection(fr.index)
    for dt in common:
        x, y = fs.loc[dt], fr.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < min_assets:
            continue
        xx, yy = x[m], y[m]
        if xx.nunique() < 3 or yy.nunique() < 3:
            continue
        r = xx.rank().corr(yy.rank())
        if not np.isnan(r):
            ics.append(r)
            dates.append(dt)
    return pd.Series(ics, index=dates)


def summarize(name, s, fwd, turnover_panel):
    ic = spearman_ic(s, fwd)
    if len(ic) < 30:
        print(f"{name:28s} n_ic={len(ic):4d}  INSUFFICIENT DATES")
        return None
    ic_mean, ic_std = ic.mean(), ic.std(ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = (ic > 0).mean()
    cov = s.notna().mean().mean() if hasattr(s, "mean") else np.nan
    # turnover: rank change of factor at 10d lag
    if turnover_panel is not None:
        common = s.index.intersection(turnover_panel.index)
        a = s.loc[common].rank(axis=1)
        b = turnover_panel.loc[common].rank(axis=1)
        m = a.notna() & b.notna()
        to = (a[m].sub(b[m]).abs().mean(axis=1)).mean() if m.any().any() else np.nan
    else:
        to = np.nan
    # subsample ICs
    def sub(lo, hi):
        sub = ic[(ic.index >= lo) & (ic.index <= hi)]
        if len(sub) < 10:
            return None
        return [round(sub.mean(), 4), round(sub.mean() / sub.std(ddof=1), 3) if sub.std(ddof=1) > 0 else 0.0, len(sub)]
    recent = sub(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-12-31"))
    sixm = sub(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-12-31"))
    print(f"{name:28s} IC={ic_mean: .4f} ICIR={icir: .3f} hit={hit: .3f} n={len(ic):4d} cov={cov: .3f} to={to: .3f} | 6m={sixm} | 1y={recent}")
    return {"name": name, "ic": ic_mean, "icir": icir, "hit": hit, "n": len(ic), "cov": cov, "to": to,
            "ic_series": ic, "six_m": sixm, "one_y": recent}


# ---------------- data ----------------
print("Loading panel ...")
panel, vpanel = build_panel(1500)
mp = macropanel(1500)
print(f"panel: {panel.shape[0]} dates x {panel.shape[1]} assets, range {panel.index.min().date()}..{panel.index.max().date()}")
print(f"macro: {mp.shape}, range {mp.index.min().date()}..{mp.index.max().date()}")

ret = panel.pct_change()
fwd = fwd_ret(panel, H)

# ---------------- candidate factors ----------------
factors = {}

# 1) trend: 60d momentum skipping 5
mom60 = panel / panel.shift(65) - 1.0
factors["mom_60d_skip5"] = mom60

# 2) 120d momentum skip 5
mom120 = panel / panel.shift(125) - 1.0
factors["mom_120d_skip5"] = mom120

# 3) 20d momentum skip 5 (shorter trend)
mom20 = panel / panel.shift(25) - 1.0
factors["mom_20d_skip5"] = mom20

# 4) vol-of-vol 20x60: 60d stdev of 20d realized vol
rv20 = ret.rolling(20).std()
volvol = rv20.rolling(60).std()
factors["vol_of_vol_20x60"] = volvol

# 5) downside vol ratio: 20d downside deviation / 20d total vol
down = ret.clip(upper=0).rolling(20).std()
vol20 = ret.rolling(20).std()
factors["downside_vol_ratio_20"] = down / vol20

# 6) range position 20d: (close - min20)/(max20 - min20)
roll_min = panel.rolling(20).min()
roll_max = panel.rolling(20).max()
factors["range_pos_20"] = (panel - roll_min) / (roll_max - roll_min)

# 7) beta vs XAU (gold) 60d
def beta_to(ref):
    out = {}
    for s in panel.columns:
        a = ret[s].align(ref, join="inner")
        r = pd.DataFrame({"a": a[0], "b": a[1]}).replace([np.inf, -np.inf], np.nan).dropna()
        cov = r["a"].rolling(60).cov(r["b"])
        var = r["b"].rolling(60).var()
        out[s] = cov / var
    return pd.DataFrame(out)

xau_beta = beta_to(ret["XAU"])
factors["xau_beta_60"] = xau_beta

# 8) beta vs BTC (crypto beta) 60d
btc_beta = beta_to(ret["BTC"])
factors["btc_beta_60"] = btc_beta

# 9) beta vs DXY 60d
dxy_r = mp["DXY"].pct_change()
dxy_beta = beta_to(dxy_r)
factors["dxy_beta_60"] = dxy_beta

# 10) beta vs VIX 60d
vix_r = mp["VIX"].pct_change()
factors["vix_beta_60"] = beta_to(vix_r)

# 11) US10Y sensitivity: 60d corr of daily rets with US10Y rets
us10y_r = ret["US10Y"]
corr_us10y = ret.corrwith(us10y_r, axis=0).T  # scalar per asset; need rolling instead
corr_roll = {}
for s in panel.columns:
    r = pd.DataFrame({"a": ret[s], "b": us10y_r}).replace([np.inf, -np.inf], np.nan)
    corr_roll[s] = r["a"].rolling(60).corr(r["b"])
factors["us10y_corr_60"] = pd.DataFrame(corr_roll)

# 12) efficiency ratio 20d: |change over 20d| / sum |daily rets|
eff = (panel.diff(20).abs()) / (ret.abs().rolling(20).sum())
factors["eff_ratio_20"] = eff

# 13) 60d realized vol (low-vol factor, negate later if needed)
rv60 = ret.rolling(60).std()
factors["vol_60"] = rv60

# 14) max drawdown 60d
def maxdd60(x):
    return (x / x.rolling(60).max() - 1.0).rolling(60).min()
factors["maxdd_60"] = maxdd60(panel)

print("\n=== FULL-SAMPLE & RECENT VALIDATION (horizon=10) ===")
results = {}
to_lag = {k: v.shift(H) for k, v in factors.items()}
for name, s in factors.items():
    r = summarize(name, s, fwd, to_lag[name])
    if r:
        results[name] = r

print("\ndone. candidate count:", len(results))