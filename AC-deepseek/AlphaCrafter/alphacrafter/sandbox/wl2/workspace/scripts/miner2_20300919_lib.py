"""Shared research library for miner_2: data loading, factor primitives, IC/ICIR evaluation.
Truncates all data at the visible date (2030-09-18) to avoid lookahead."""
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]
VISIBLE = "2030-09-18"

def load_prices(visible=VISIBLE):
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(visible)].set_index("date").sort_index()
        closes[a] = df["close"]
    px = pd.DataFrame(closes).dropna(how="all")
    ret = px.pct_change()
    return px, ret

def load_macro(visible=VISIBLE):
    out = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(visible)].set_index("date").sort_index()
        out[m] = df["close"]
    return pd.DataFrame(out)

def fwd_ret(ret, h):
    """Forward h-day return: ret of day t+1..t+h (skip 0)."""
    return ret.shift(-h).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)

def fwd_ret_skip(ret, h, skip):
    """Forward h-day return starting `skip` days after t."""
    return ret.shift(-(h + skip)).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)

def daily_ic(factor_df, fwd, min_obs=8):
    """Cross-sectional Spearman IC per date."""
    dates = factor_df.index.intersection(fwd.index)
    ics = []
    for d in dates:
        f = factor_df.loc[d]
        r = fwd.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < min_obs:
            continue
        ics.append((d, f[mask].rank().corr(r[mask].rank())))
    return pd.Series(dict(ics))

def eval_factor(factor_df, ret, horizon=10, skip=0, min_obs=8, name=""):
    fwd = fwd_ret_skip(ret, horizon, skip) if skip else fwd_ret(ret, horizon)
    ic = daily_ic(factor_df, fwd, min_obs).dropna()
    n = len(ic)
    if n < 60:
        return {"name": name, "n_dates": n, "ic": np.nan, "icir": np.nan}
    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    icir = ic_mean / ic_std * np.sqrt(n) if ic_std > 0 else np.nan
    hit = (ic > 0).mean()
    rnk = factor_df.rank(axis=1).dropna(how="all")
    turn = (rnk.diff().abs() / (len(ASSETS) - 1)).mean().mean() if len(rnk) > 2 else np.nan
    cov = factor_df.notna().mean().mean()
    cov8 = (factor_df.notna().sum(axis=1) >= 8).mean()
    return {"name": name, "n_dates": n, "ic": round(float(ic_mean), 4), "icir": round(float(icir), 3),
            "hit": round(float(hit), 3), "turnover": round(float(turn), 3) if turn == turn else np.nan,
            "coverage": round(float(cov), 3), "cov_dates_ge8": round(float(cov8), 3)}

def consec_max(vals):
    mx = 0; cur = 0
    for v in vals:
        cur = cur + 1 if v else 0
        mx = max(mx, cur)
    return mx

def library_signals(px, ret):
    """Recompute a handful of library factor signals for correlation reference (ranked)."""
    sig = {}
    def ranknorm(s):
        return s.rank(axis=1)
    mom20 = px.pct_change(20).shift(5)
    mom60 = px.pct_change(60).shift(5)
    sig["mom20_volproxy60"] = ranknorm(mom20 / (1 + mom60.abs()))
    spx = ret["SPX"]
    db = pd.DataFrame(index=ret.index, columns=px.columns, dtype=float)
    for a in px.columns:
        r = ret[a]
        db[a] = r.rolling(60).apply(
            lambda x: (np.cov(x, spx.loc[x.index])[0, 1] / np.var(spx.loc[x.index])
                       if len(x) >= 15 and np.var(spx.loc[x.index]) > 0 else np.nan), raw=False)
    sig["downbeta_spx_60"] = ranknorm(db)
    up = (ret > 0).astype(int)
    mcg = up.rolling(20).apply(lambda x: consec_max(x), raw=True)
    sig["max_consec_gain_20"] = ranknorm(mcg)
    dsh = px.rolling(60).apply(lambda x: np.argmax(x[::-1]) if len(x) == 60 else np.nan, raw=True).shift(1)
    sig["days_since_high_60"] = ranknorm(dsh)
    sc = pd.DataFrame(index=ret.index, columns=px.columns, dtype=float)
    for a in px.columns:
        sc[a] = ret[a].rolling(60).corr(spx)
    sig["spx_corr60"] = ranknorm(sc)
    vol20 = ret.rolling(20).std()
    sig["calmness_20"] = ranknorm(-vol20)
    return sig
