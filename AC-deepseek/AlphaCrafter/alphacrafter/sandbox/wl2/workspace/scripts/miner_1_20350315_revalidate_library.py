"""miner_1 library re-validation with data through 2035-03-14 (visible_through).
Recomputes all live library factor signals per-asset on own calendar, computes
10d-horizon cross-sectional Spearman IC / ICIR over full sample + regime slices,
and checks the admission gate |IC|>=0.007, |ICIR|>=0.084.
No lookahead: factor at t uses data <= t; forward return uses t+1..t+h.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE = "2035-03-14"
TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")
MIN_ASSETS = 8


def load_asset(symbol):
    p = (INDEX_DIR if symbol in MACRO else DATA_DIR) / f"{symbol}.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date").reset_index(drop=True)
    return df


def load_ohlcv():
    out = {}
    for a in TRADABLES:
        df = load_asset(a)
        out[a] = pd.DataFrame({
            "open": df["open"].astype(float).values,
            "high": df["high"].astype(float).values,
            "low": df["low"].astype(float).values,
            "close": df["close"].astype(float).values,
        }, index=pd.to_datetime(df["date"]))
    return out


def macro_series(name):
    df = load_asset(name)
    return pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"]), name=name)


def per_asset(ohlcv, func):
    """Apply func to each asset's own-calendar dataframe/series, reindex to union index."""
    out = {}
    for a in ohlcv:
        out[a] = func(ohlcv[a]).reindex(union_index)
    return pd.DataFrame(out, index=union_index)


def fwd_ret(s, h):
    return s.shift(-h) / s - 1.0


def compute_ic(F, R):
    dates = F.index.intersection(R.index)
    Fr = F.loc[dates].rank(axis=1).values
    Rr = R.loc[dates].rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= MIN_ASSETS
    ics = np.full(len(dates), np.nan)
    idx = np.where(valid)[0]
    for i in idx:
        f = Fr[i, m[i]] - Fr[i, m[i]].mean()
        r = Rr[i, m[i]] - Rr[i, m[i]].mean()
        denom = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / denom if denom > 0 else np.nan
    return pd.Series(ics, index=dates)


def rolling_beta(a_ret, m_ret, win, min_obs):
    cov = a_ret.rolling(win, min_periods=min_obs).cov(m_ret)
    var = m_ret.rolling(win, min_periods=min_obs).var()
    return cov / var


def build_library_signals(ohlcv):
    sig = {}
    union = union_index

    def s_close(d): return d["close"]
    def s_ret(d): return d["close"].pct_change()

    # ---- momentum family ----
    for label, num, den in [("mom_10d_skip5", 5, 15), ("mom_20d_skip5", 5, 25),
                            ("mom_120d_skip5", 5, 125), ("mom_180d_skip5", 5, 185)]:
        sig[label] = per_asset(ohlcv, lambda d, n=num, m=den: d["close"].shift(n) / d["close"].shift(m) - 1.0)
    sig["mom20_volproxy60"] = per_asset(ohlcv, lambda d: d["close"].shift(5) / d["close"].shift(25) - 1.0)
    sig["mom30_vol60"] = per_asset(ohlcv, lambda d: (d["close"].shift(5) / d["close"].shift(35) - 1.0)
                                   / d["close"].pct_change().rolling(60, min_periods=15).std())
    sig["vol_of_vol20x60"] = per_asset(ohlcv, lambda d: d["close"].pct_change().rolling(20, min_periods=5).std()
                                       .rolling(60, min_periods=15).std())
    # ---- volatility / regime ----
    sig["calmness_20"] = per_asset(ohlcv, lambda d: (d["close"].pct_change().abs()
                                                     < 0.5 * d["close"].pct_change().rolling(20, min_periods=10).std())
                                   .rolling(20, min_periods=10).mean())
    sig["volcluster_60"] = per_asset(ohlcv, lambda d: d["close"].pct_change().abs().rolling(60, min_periods=40)
                                     .corr(d["close"].pct_change().abs().shift(1)))
    # ---- intraday / position ----
    sig["intraday_drift_20"] = per_asset(ohlcv, lambda d: (d["close"] / d["open"] - 1.0).rolling(20, min_periods=10).mean())
    sig["close_pos_20"] = per_asset(ohlcv, lambda d: ((d["close"] - d["low"]) / (d["high"] - d["low"]))
                                   .replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=10).mean())
    sig["gain_loss_20"] = per_asset(ohlcv, lambda d: d["close"].pct_change().clip(lower=0).rolling(20, min_periods=10).mean()
                                    / d["close"].pct_change().clip(upper=0).abs().rolling(20, min_periods=10).mean())
    # ---- days since high ----
    def days_since_high(d):
        c = d["close"]
        out = pd.Series(np.nan, index=c.index)
        for i in range(59, len(c)):
            win = c.iloc[i - 59:i + 1]
            mx = win.max()
            if mx != mx:
                continue
            idx = win.index[win == mx][-1]
            out.iloc[i] = (c.index[i] - idx).days
        return out
    sig["days_since_high_60"] = per_asset(ohlcv, days_since_high)
    # ---- max consecutive gains/losses (21d) ----
    def max_consec(d, up=True):
        r = (d["close"].pct_change() > 0).astype(int)
        if not up:
            r = (d["close"].pct_change() < 0).astype(int)
        out = pd.Series(0.0, index=r.index)
        run = 0
        vals = r.values
        for i in range(len(vals)):
            run = run + 1 if vals[i] == 1 else 0
            out.iloc[i] = run
        return out.rolling(21, min_periods=1).max()
    sig["max_consec_gain_20"] = per_asset(ohlcv, lambda d: max_consec(d, True))
    sig["max_consec_loss_20"] = per_asset(ohlcv, lambda d: max_consec(d, False))
    # ---- range position ----
    sig["range_pos_252"] = per_asset(ohlcv, lambda d: (d["close"] - d["close"].rolling(252, min_periods=30).min())
                                     / (d["close"].rolling(252, min_periods=30).max()
                                        - d["close"].rolling(252, min_periods=30).min()))
    # ---- SPX-related ----
    spx_ret = ohlcv["SPX"]["close"].pct_change()
    sig["spx_corr60"] = per_asset(ohlcv, lambda d: d["close"].pct_change().rolling(60, min_periods=15)
                                  .corr(spx_ret.reindex(d.index)))
    sig["lagbeta_spx_60"] = per_asset(ohlcv, lambda d: rolling_beta(d["close"].pct_change(),
                                                                    spx_ret.reindex(d.index).shift(1), 60, 15))
    # downside beta: SPX down days
    def downbeta(d):
        a = d["close"].pct_change()
        m = spx_ret.reindex(d.index)
        both = pd.concat([a.rename("a"), m.rename("m")], axis=1)
        both["m_neg"] = (both["m"] < 0).astype(float).replace(0.0, np.nan)
        b = (both["a"] * both["m_neg"]).rolling(60, min_periods=15).sum() / \
            (both["m_neg"] ** 2).rolling(60, min_periods=15).sum()
        return b
    sig["downbeta_spx_60"] = per_asset(ohlcv, downbeta)
    # ---- macro-conditional betas ----
    dxy = macro_series("DXY").pct_change()
    usdjpy = macro_series("USDJPY").pct_change()
    vix = macro_series("VIX").pct_change()
    vix_close = macro_series("VIX")
    dxy_close = macro_series("DXY")
    usdjpy_close = macro_series("USDJPY")

    def beta_cond(d, macro_ret, macro_mom):
        a = d["close"].pct_change()
        m = macro_ret.reindex(d.index)
        b = rolling_beta(a, m, 60, 15)
        return b * macro_mom.reindex(d.index)
    sig["dxy_beta_cond_60x20"] = per_asset(ohlcv, lambda d: beta_cond(d, dxy, dxy_close / dxy_close.shift(20) - 1.0))
    sig["usdjpy_beta_cond_120x60"] = per_asset(ohlcv, lambda d: rolling_beta(d["close"].pct_change(),
                                                                             usdjpy.reindex(d.index), 120, 30)
                                              * (usdjpy_close.reindex(d.index) / usdjpy_close.reindex(d.index).shift(60) - 1.0))
    sig["vix_beta_cond_60x20"] = per_asset(ohlcv, lambda d: -beta_cond(d, vix, vix_close / vix_close.shift(20) - 1.0))
    return sig


def regime_ic(ic_series):
    out = {}
    for lo, hi in [(2020, 2021), (2022, 2022), (2023, 2024), (2025, 2026),
                   (2027, 2028), (2029, 2030), (2031, 2032), (2033, 2034), (2035, 2035)]:
        g = ic_series[(ic_series.index.year >= lo) & (ic_series.index.year <= hi)]
        if len(g) > 10:
            out[f"{lo}-{hi}"] = (round(float(g.mean()), 4), round(float(g.mean() / g.std()), 3) if g.std() > 0 else 0.0, len(g))
    return out


def main():
    global union_index
    ohlcv = load_ohlcv()
    union_index = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in ohlcv.values()])))
    # forward returns per asset on own calendar
    fwd = {}
    for h in (1, 2, 3, 5, 10, 20):
        fwd[h] = per_asset(ohlcv, lambda d, hh=h: fwd_ret(d["close"], hh))
    fwd10 = fwd[10]

    print(f"Visible through {VISIBLE}; assets={len(TRADABLES)}; union dates={len(union_index)}")
    sig = build_library_signals(ohlcv)
    print(f"library factors: {len(sig)}")

    rows = []
    for label, F in sig.items():
        ic10 = compute_ic(F, fwd10).dropna()
        ic = float(ic10.mean())
        icir = float(ic10.mean() / ic10.std()) if ic10.std() > 0 else 0.0
        hit = float((ic10 > 0).mean())
        last252 = ic10.tail(252)
        last126 = ic10.tail(126)
        last_ic = float(last252.mean()) if len(last252) > 10 else float("nan")
        last_icir = float(last252.mean() / last252.std()) if len(last252) > 10 and last252.std() > 0 else float("nan")
        last126_ic = float(last126.mean()) if len(last126) > 10 else float("nan")
        last126_icir = float(last126.mean() / last126.std()) if len(last126) > 10 and last126.std() > 0 else float("nan")
        cov = float(F.notna().sum().sum() / (F.shape[0] * F.shape[1]))
        # turnover 10d
        rk = F.rank(axis=1, pct=True)
        to_vals = []
        for i in range(10, len(rk), 10):
            a, b = rk.iloc[i - 10], rk.iloc[i]
            m = a.notna() & b.notna()
            if m.sum() >= MIN_ASSETS:
                to_vals.append(float((b[m] - a[m]).abs().mean()))
        to = float(np.mean(to_vals)) if to_vals else float("nan")
        passed = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
        rows.append((label, ic, icir, hit, len(ic10), cov, to, last_ic, last_icir, last126_ic, last126_icir, passed))
        print(f"{label:28s} IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(ic10):4d} "
              f"cov={cov:.3f} to={to:.3f} last252_ic={last_ic:+.4f} last252_icir={last_icir:+.3f} "
              f"last126_ic={last126_ic:+.4f} last126_icir={last126_icir:+.3f} {'PASS' if passed else 'fail'}")

    print("\n=== regime IC (full-sample factor, 10d horizon) ===")
    for label in ["max_consec_gain_20", "mom_180d_skip5", "range_pos_252", "spx_corr60",
                  "downbeta_spx_60", "calmness_20", "volcluster_60", "mom20_volproxy60"]:
        F = sig[label]
        ic10 = compute_ic(F, fwd10).dropna()
        print(f"\n[{label}]")
        for k, (i, ir, n) in regime_ic(ic10).items():
            print(f"  {k}: ic={i:+.4f} icir={ir:+.3f} n={n}")

    import json
    out = {r[0]: {"ic": r[1], "icir": r[2], "hit": r[3], "n_ic_dates": r[4], "coverage": r[5],
                  "turnover_10d_rank": r[6], "last252_ic": r[7], "last252_icir": r[8],
                  "last126_ic": r[9], "last126_icir": r[10], "gate_pass": r[11]}
           for r in rows}
    with open("scripts/miner_1_20350315_revalidate_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nsaved scripts/miner_1_20350315_revalidate_results.json")


if __name__ == "__main__":
    main()
