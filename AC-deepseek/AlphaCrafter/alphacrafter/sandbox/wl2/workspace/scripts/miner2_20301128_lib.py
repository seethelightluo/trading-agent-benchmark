"""Shared research library for miner_2 (2030-11-28 cycle).
Truncates all data at the visible date (2030-11-27) to avoid lookahead.
"""
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]
VISIBLE = "2030-11-27"

GATE_IC = 0.0070
GATE_ICIR = 0.0840


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
    return ret.shift(-(h + skip)).rolling(h).apply(lambda x: np.prod(1 + x) - 1, raw=True)


def daily_ic(factor_df, fwd, min_obs=8):
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
        return {"name": name, "n_dates": n, "ic": np.nan, "icir": np.nan, "ok": False}
    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    icir = ic_mean / ic_std * np.sqrt(n) if ic_std > 0 else np.nan
    hit = (ic > 0).mean()
    rnk = factor_df.rank(axis=1).dropna(how="all")
    turn = (rnk.diff().abs() / (len(ASSETS) - 1)).mean().mean() if len(rnk) > 2 else np.nan
    cov = factor_df.notna().mean().mean()
    cov8 = (factor_df.notna().sum(axis=1) >= 8).mean()
    ok = (abs(ic_mean) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    return {"name": name, "n_dates": n, "ic": round(float(ic_mean), 4),
            "icir": round(float(icir), 3), "hit": round(float(hit), 3),
            "turnover": round(float(turn), 3) if turn == turn else np.nan,
            "coverage": round(float(cov), 3), "cov_dates_ge8": round(float(cov8), 3),
            "ok": bool(ok)}


def decay_analysis(factor_df, ret, horizons=(1, 2, 3, 5, 10, 20), min_obs=8, name=""):
    out = {}
    for h in horizons:
        fwd = fwd_ret(ret, h)
        ic = daily_ic(factor_df, fwd, min_obs).dropna()
        if len(ic) >= 60:
            out[h] = round(float(ic.mean()), 4)
        else:
            out[h] = None
    return out


def regime_ic(factor_df, ret, horizon=10, min_obs=8):
    """IC by sub-period to assess regime robustness."""
    fwd = fwd_ret(ret, horizon)
    ic = daily_ic(factor_df, fwd, min_obs).dropna()
    bounds = [(ic.index.min(), "2021-12-31"), ("2022-01-01", "2024-12-31"),
              ("2025-01-01", ic.index.max())]
    labels = ["2020-2021", "2022-2024", "2025-2030"]
    out = {}
    for (lo, hi), lab in zip(bounds, labels):
        sub = ic[(ic.index >= pd.Timestamp(lo)) & (ic.index <= pd.Timestamp(hi))]
        if len(sub) >= 30:
            out[lab] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sub.std(ddof=1) * np.sqrt(len(sub))), 3)
                        if sub.std(ddof=1) > 0 else None,
                        "n": int(len(sub))}
        else:
            out[lab] = {"ic": None, "icir": None, "n": int(len(sub))}
    out["last250"] = {"ic": round(float(ic.tail(250).mean()), 4), "n": int(len(ic.tail(250)))}
    return out


def library_signals(px, ret):
    """Recompute key library factor signals (ranked) for correlation reference."""
    sig = {}
    def ranknorm(s):
        return s.rank(axis=1)
    sig["mom20_volproxy60"] = ranknorm(px.pct_change(20).shift(5) / (1 + px.pct_change(60).shift(5).abs()))
    sig["mom_180d_skip5"] = ranknorm(px.shift(5) / px.shift(185) - 1)
    sig["range_pos_252"] = ranknorm((px - px.rolling(252).min()) / (px.rolling(252).max() - px.rolling(252).min()))
    spx = ret["SPX"]
    db = pd.DataFrame(index=ret.index, columns=px.columns, dtype=float)
    for a in px.columns:
        r = ret[a]
        db[a] = r.rolling(60).apply(
            lambda x: (np.cov(x, spx.loc[x.index])[0, 1] / np.var(spx.loc[x.index])
                       if len(x) >= 15 and np.var(spx.loc[x.index]) > 0 else np.nan), raw=False)
    sig["downbeta_spx_60"] = ranknorm(db)
    sc = pd.DataFrame(index=ret.index, columns=px.columns, dtype=float)
    for a in px.columns:
        sc[a] = ret[a].rolling(60).corr(spx)
    sig["spx_corr60"] = ranknorm(sc)
    up = (ret > 0).astype(int)
    def consec_max(vals):
        mx = 0; cur = 0
        for v in vals:
            cur = cur + 1 if v else 0
            mx = max(mx, cur)
        return mx
    mcg = up.rolling(20).apply(lambda x: consec_max(x), raw=True)
    sig["max_consec_gain_20"] = ranknorm(mcg)
    return sig


def corr_with_library(factor_df, px, ret):
    """Max abs cross-sectional-rank correlation of candidate with library factors."""
    sig = library_signals(px, ret)
    rnk = factor_df.rank(axis=1)
    best = None
    best_corr = 0.0
    for k, v in sig.items():
        common = rnk.index.intersection(v.index)
        cs = []
        for d in common:
            a = rnk.loc[d]; b = v.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= 8:
                cs.append(a[m].corr(b[m]))
        if len(cs) >= 60:
            rho = float(np.mean(cs))
            if abs(rho) > abs(best_corr):
                best_corr = rho
                best = k
    return round(best_corr, 4), best
