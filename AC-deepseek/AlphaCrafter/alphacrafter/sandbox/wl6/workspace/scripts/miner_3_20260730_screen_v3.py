"""miner_3 screen v3: fresh candidate factor families.

Focus on trend-quality, volatility-term-structure, volume/flow, intraday-vs-gap,
and cross-asset linkage factors that were NOT in screen v1/v2.
Horizon-10 cross-sectional Spearman rank IC preview with gate check
(|IC|>=0.007, |ICIR|>=0.084). Data truncated at 2026-07-30.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_utils import (load_close, load_panel, forward_returns, rank_ic_series,
                          summarize_ic, DATA_DIR, INDEX_DIR, TRADABLES, OBSERVABLES)

px, vol = load_panel()
fwd10 = forward_returns(px, 10)
pxr = px.pct_change()

# EW index of tradables (per-asset calendar, then aligned)
def per_asset(func, symbols=None):
    out = {}
    for s in (symbols or TRADABLES):
        df = load_close(s)
        out[s] = func(df)
    return pd.DataFrame(out).reindex(px.index)


def macro_close(name):
    df = load_close(name, INDEX_DIR)
    return df["close"].astype(float)


def eff_ratio(n):
    """Kaufman efficiency ratio: |c - c.shift(n)| / sum(|r|, n)."""
    def f(df):
        c = df["close"]
        r = c.pct_change().abs()
        return (c - c.shift(n)).abs() / r.rolling(n).sum()
    return per_asset(f)


def win_rate(n):
    def f(df):
        r = df["close"].pct_change()
        return (r > 0).rolling(n).mean()
    return per_asset(f)


def updown_asym(n):
    """Sum of positive returns / sum of |negative returns| over window."""
    def f(df):
        r = df["close"].pct_change()
        pos = r.clip(lower=0).rolling(n).sum()
        neg = (-r.clip(upper=0)).rolling(n).sum().replace(0, np.nan)
        return pos / neg
    return per_asset(f)


def vol_ratio(a, b):
    def f(df):
        r = df["close"].pct_change()
        return r.rolling(a).std() / r.rolling(b).std()
    return per_asset(f)


def ewma_vol_ratio(span_a, span_b):
    def f(df):
        r = df["close"].pct_change()
        va = r.ewm(span=span_a).std()
        vb = r.ewm(span=span_b).std()
        return va / vb
    return per_asset(f)


def volume_z(n, base=60):
    def f(df):
        if "volume" not in df:
            return pd.Series(np.nan, index=df.index)
        v = df["volume"].astype(float).replace(0, np.nan)
        mu = v.rolling(base).mean()
        sd = v.rolling(base).std()
        return (v - mu) / sd.replace(0, np.nan)
    return per_asset(f)


def volume_trend(a, b):
    def f(df):
        if "volume" not in df:
            return pd.Series(np.nan, index=df.index)
        v = df["volume"].astype(float).replace(0, np.nan)
        return v.rolling(a).mean() / v.rolling(b).mean()
    return per_asset(f)


def price_vol_corr(n):
    def f(df):
        r = df["close"].pct_change()
        v = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
        return r.rolling(n).corr(v)
    return per_asset(f)


def gap_avg(n):
    """Mean |open/prev_close - 1| over window (overnight gap size)."""
    def f(df):
        c = df["close"].shift(1)
        g = (df["open"] / c - 1.0).abs()
        return g.rolling(n).mean()
    return per_asset(f)


def intraday_strength(n):
    """Mean close/open - 1 over window (intraday direction)."""
    def f(df):
        g = df["close"] / df["open"] - 1.0
        return g.rolling(n).mean()
    return per_asset(f)


def stoch_pos(n):
    """(c - min(low,n)) / (max(high,n) - min(low,n))."""
    def f(df):
        lo = df["low"].rolling(n).min()
        hi = df["high"].rolling(n).max()
        rng = (hi - lo).replace(0, np.nan)
        return (df["close"] - lo) / rng
    return per_asset(f)


def days_since_high(n):
    """Negative count of days since rolling max close (higher = fresher high)."""
    def f(df):
        c = df["close"]
        roll_max = c.rolling(n, min_periods=n // 2).max()
        # count days since the max was achieved
        idx = c.groupby(roll_max).cumcount()
        out = pd.Series(index=df.index, dtype=float)
        # simpler: distance from max
        out = c / roll_max - 1.0
        return out
    return per_asset(f)


def beta_ew(win):
    """Rolling beta of asset vs equal-weight tradable index (per-asset calendar)."""
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        ew = pxr.mean(axis=1).reindex(df.index).ffill()
        cov = r.rolling(win).cov(ew)
        var = ew.rolling(win).var().replace(0, np.nan)
        out[s] = cov / var
    return pd.DataFrame(out).reindex(px.index)


def downside_beta_ew(win):
    """Beta vs EW index computed only on down days of the EW index."""
    out = {}
    ew = pxr.mean(axis=1)
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        ews = ew.reindex(df.index).ffill()
        mask = ews < 0
        betas = pd.Series(np.nan, index=df.index)
        rm = r.rolling(win).apply(
            lambda x: np.cov(x, ews.loc[x.index])[0, 1] / np.var(ews.loc[x.index]) if np.var(ews.loc[x.index]) > 1e-12 else np.nan,
            raw=False)
        betas = rm
        out[s] = betas
    return pd.DataFrame(out).reindex(px.index)


def alpha_ew(win):
    """Idiosyncratic return vs EW index: mean(r - beta*r_ew) over window."""
    out = {}
    ew = pxr.mean(axis=1)
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        ews = ew.reindex(df.index).ffill()
        beta = r.rolling(win).cov(ews) / ews.rolling(win).var().replace(0, np.nan)
        alpha = (r - beta * ews).rolling(win).mean()
        out[s] = alpha
    return pd.DataFrame(out).reindex(px.index)


def corr_with(win, mname):
    m = macro_close(mname)
    mr = m.pct_change()
    out = {}
    for s in TRADABLES:
        df = load_close(s)
        r = df["close"].pct_change()
        mrs = mr.reindex(df.index).ffill()
        out[s] = r.rolling(win).corr(mrs)
    return pd.DataFrame(out).reindex(px.index)


def max_ret(n):
    def f(df):
        return df["close"].pct_change().rolling(n).max()
    return per_asset(f)


def min_ret(n):
    def f(df):
        return df["close"].pct_change().rolling(n).min()
    return per_asset(f)


def clv_diff(a, b):
    """Change in intraday positioning: CLV(a) - CLV(b)."""
    def f(df):
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        clv = (df["close"] - df["low"]) / rng
        return clv.rolling(a).mean() - clv.rolling(b).mean()
    return per_asset(f)


candidates = {
    "eff_ratio_20d": eff_ratio(20),
    "eff_ratio_60d": eff_ratio(60),
    "win_rate_20d": win_rate(20),
    "win_rate_60d": win_rate(60),
    "updown_asym_20d": updown_asym(20),
    "updown_asym_60d": updown_asym(60),
    "vol_ratio_5x60": vol_ratio(5, 60),
    "vol_ratio_10x60": vol_ratio(10, 60),
    "ewma_vol_ratio_10x60": ewma_vol_ratio(10, 60),
    "volume_z_20x60": volume_z(20, 60),
    "volume_trend_5x60": volume_trend(5, 60),
    "price_vol_corr_20d": price_vol_corr(20),
    "gap_avg_20d": gap_avg(20),
    "intraday_strength_20d": intraday_strength(20),
    "stoch_pos_60d": stoch_pos(60),
    "dist_high_60d": days_since_high(60),
    "beta_ew_60d": beta_ew(60),
    "downside_beta_ew_60d": downside_beta_ew(60),
    "alpha_ew_60d": alpha_ew(60),
    "corr_dxy_60d": corr_with(60, "DXY"),
    "corr_vix_60d": corr_with(60, "VIX"),
    "max_ret_20d": max_ret(20),
    "min_ret_20d": min_ret(20),
    "clv_diff_10x60": clv_diff(10, 60),
}

print(f"panel dates: {len(px)}, instruments: {len(px.columns)}")
print(f"{'factor':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}  gate")
results = {}
for name, f in candidates.items():
    f = f.reindex(px.index)
    s = rank_ic_series(f, fwd10)
    res = summarize_ic(s, name, 10)
    results[name] = res
    flag = "PASS" if res["pass_gate"] else ""
    print(f"{name:<24}{res['ic']:>8.4f}{res['icir']:>8.4f}{res['ic_hit_ratio']:>7.3f}{res['n_ic_dates']:>6d}  {flag}")

print("\nPASSED:")
for name, res in results.items():
    if res["pass_gate"]:
        print(f"  {name}: ic={res['ic']:.4f} icir={res['icir']:.4f} hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']}")
