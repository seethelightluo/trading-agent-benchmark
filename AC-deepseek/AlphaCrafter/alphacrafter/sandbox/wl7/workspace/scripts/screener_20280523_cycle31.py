"""
Screener cycle 2028-05-23 (data thru ~2028-05-22).
Reuse miner round-30 pipeline: load 15-asset panel + macro, compute regime
stats and re-validate the 8 EFFECTIVE library factors on warm/live/recent
windows (recent250d + recent60d) to drive the quality_ic_tilt ensemble.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8
WARM_END = "2026-07-15"
LIVE_START = "2026-07-16"
DAYS = 4000


def load_ohlcv(days=DAYS):
    closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
    for s in WATCH:
        df = get_stock_daily_data(s, days=days)
        if df is None or not len(df):
            continue
        df = df.set_index("date")
        closes[s] = df["close"].astype(float)
        opens[s] = df["open"].astype(float)
        highs[s] = df["high"].astype(float)
        lows[s] = df["low"].astype(float)
        vols[s] = df["volume"].astype(float)

    def _p(d):
        p = pd.concat(d, axis=1, sort=True)
        return p[~p.index.duplicated(keep="last")].sort_index()
    return _p(closes), _p(opens), _p(highs), _p(lows), _p(vols)


def load_macro():
    out = {}
    for s in MACRO:
        df = get_index_daily_data(s, days=DAYS)
        if df is not None and len(df):
            out[s] = df.set_index("date")["close"].astype(float)
    return out


def per_asset(fn):
    def wrapper(panel):
        cols = {}
        for a in panel.columns:
            s = panel[a].dropna()
            cols[a] = fn(s)
        return pd.DataFrame(cols, index=panel.index)
    return wrapper


def fwd_returns(panel, h):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor, fwd):
    f = factor.stack().rename("f")
    r = fwd.stack().rename("r")
    j = pd.concat([f, r], axis=1).dropna()
    if len(j) == 0:
        return pd.Series(dtype=float)
    j["fr"] = j.groupby(level=0)["f"].rank()
    j["rr"] = j.groupby(level=0)["r"].rank()
    cnt = j.groupby(level=0).size()
    keep = cnt[cnt >= MIN_ASSETS].index
    j = j[j.index.get_level_values(0).isin(keep)]
    g = j.groupby(level=0)
    n = g.size()
    sx, sy = g["fr"].sum(), g["rr"].sum()
    sxx = g["fr"].apply(lambda s: float((s ** 2).sum()))
    syy = g["rr"].apply(lambda s: float((s ** 2).sum()))
    sxy = g.apply(lambda d: float((d["fr"] * d["rr"]).sum()))
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
    ic = num / den
    return ic.sort_index()


def turnover_10d_rank(factor):
    ranks = factor.rank(axis=1)
    out, dates = [], ranks.index
    for i in range(10, len(dates)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            out.append(float((a[both] - b[both]).abs().mean()))
    return float(np.mean(out)) if out else float("nan")


def library_signals(close, high, low, vol, macro):
    lib = {}
    r = close.pct_change()
    m20 = per_asset(lambda s: s.shift(5) / s.shift(25) - 1.0)(close)
    lib["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    ew = close.mean(axis=1)
    ew_r = ew.pct_change()

    def ew_beta(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    lib["beta_ew_60d"] = per_asset(ew_beta)(close)

    def dsvr(s):
        rr = s.pct_change()
        down = rr.where(rr < 0, 0.0)
        ds = np.sqrt((down ** 2).rolling(20).mean())
        tot = rr.rolling(20).std()
        return -(ds / tot)
    lib["downside_vol_ratio_20"] = per_asset(dsvr)(close)
    lib["max_ret_20d"] = r.rolling(20).max()

    def ew_corr(s):
        z = pd.concat([s.pct_change().rename("r"), ew_r.rename("m")], axis=1).dropna()
        return z["r"].rolling(60).corr(z["m"])
    lib["corr_ew_60"] = per_asset(ew_corr)(close)

    def fx_cond(ref):
        ref20 = (ref / ref.shift(20) - 1.0)

        def f(s):
            z = pd.concat([s.pct_change().rename("r"), ref.pct_change().reindex(s.index).rename("x")], axis=1).dropna()
            beta = z["r"].rolling(60).cov(z["x"]) / z["x"].rolling(60).var()
            return beta * ref20.reindex(s.index)
        return per_asset(f)(close)
    lib["dxy_beta_cond_60x20"] = fx_cond(macro["DXY"].dropna())
    lib["eurusd_beta_cond_60x20"] = fx_cond(macro["EURUSD"].dropna())

    def kurt(s):
        rr = s.pct_change().shift(5)
        return rr.rolling(20, min_periods=12).kurt()
    lib["kurt_20d_skip5"] = per_asset(kurt)(close)
    return lib


if __name__ == "__main__":
    close, open_, high, low, vol = load_ohlcv()
    macro = load_macro()
    libsig = library_signals(close, high, low, vol, macro)
    print(f"panel: {close.shape[0]} dates x {close.shape[1]} assets; data end {close.index[-1].date()}")

    last = close.index[-1]
    r = close.pct_change()
    ew = close.mean(axis=1)

    def ret(s, n):
        return close[s].iloc[-1] / close[s].iloc[-1 - n] - 1.0

    print("\n===== REGIME =====")
    for n in (5, 10, 20, 60):
        print(f"EW mkt {n}d = {ew.iloc[-1]/ew.iloc[-1-n]-1:+.2%}")
    vix = macro["VIX"]
    print(f"VIX last={vix.iloc[-1]:.2f} 5d={vix.iloc[-1]/vix.iloc[-5]-1:+.1%} 20d={vix.iloc[-1]/vix.iloc[-21]-1:+.1%} 60d={vix.iloc[-1]/vix.iloc[-61]-1:+.1%}")
    dxy = macro["DXY"]
    print(f"DXY last={dxy.iloc[-1]:.2f} 20d={dxy.iloc[-1]/dxy.iloc[-21]-1:+.1%} 60d={dxy.iloc[-1]/dxy.iloc[-61]-1:+.1%}")
    eurusd = macro["EURUSD"]
    print(f"EURUSD last={eurusd.iloc[-1]:.4f} 20d={eurusd.iloc[-1]/eurusd.iloc[-21]-1:+.1%} 60d={eurusd.iloc[-1]/eurusd.iloc[-61]-1:+.1%}")
    usdjpy = macro["USDJPY"]
    print(f"USDJPY last={usdjpy.iloc[-1]:.2f} 20d={usdjpy.iloc[-1]/usdjpy.iloc[-21]-1:+.1%} 60d={usdjpy.iloc[-1]/usdjpy.iloc[-61]-1:+.1%}")

    # SPX realized vol
    spx_r = close["SPX"].pct_change()
    print(f"SPX rvol20d = {spx_r.iloc[-20:].std()*np.sqrt(252):.1%} ann | SPX 20d={ret('SPX',20):+.1%} 60d={ret('SPX',60):+.1%}")

    # pairwise corr 60d (mean of pairwise rolling corr)
    rr = r.iloc[-60:]
    c = rr.corr()
    vals = c.values[np.triu_indices(len(c), k=1)]
    print(f"mean pairwise 60d corr = {np.nanmean(vals):+.3f}")

    # dispersion
    r20 = close.iloc[-1] / close.iloc[-21] - 1.0
    r60 = close.iloc[-1] / close.iloc[-61] - 1.0
    print(f"20d max-min spread = {(r20.max()-r20.min())*100:.1f}pp")

    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    lastc = close.iloc[-1]
    print("\n===== PER-ASSET (thru %s) =====" % last.date())
    print(f"{'asset':10s} {'20d':>8s} {'60d':>8s} {'>MA20':>6s} {'>MA60':>6s}")
    for a in WATCH:
        if a in close.columns:
            print(f"{a:10s} {r20[a]:+8.1%} {r60[a]:+8.1%} {str(lastc[a]>ma20[a]):>6s} {str(lastc[a]>ma60[a]):>6s}")
    # frozen feed check
    print("\nfrozen feeds (zero 20d/60d ret):",
          [a for a in WATCH if a in close.columns and abs(r20[a]) < 1e-9 and abs(r60[a]) < 1e-9])

    print("\n===== LIBRARY RE-VALIDATION (drift + recent tilt) =====")
    fwd10 = fwd_returns(close, 10)
    recent_start_250 = last - pd.Timedelta(days=365)
    recent_start_60 = last - pd.Timedelta(days=90)
    rows = []
    for fid, sig in libsig.items():
        warm = sig.loc[:pd.Timestamp(WARM_END)]
        ic_w = rank_ic_series(warm, fwd10)
        icw = float(ic_w.mean()) if len(ic_w) else float("nan")
        icirw = float(ic_w.mean() / ic_w.std()) if len(ic_w) > 2 and ic_w.std() > 0 else float("nan")
        ic_l = rank_ic_series(sig.loc[LIVE_START:], fwd10)
        icl = float(ic_l.mean()) if len(ic_l) else float("nan")
        icirl = float(ic_l.mean() / ic_l.std()) if len(ic_l) > 2 and ic_l.std() > 0 else float("nan")
        ic_r250 = rank_ic_series(sig.loc[recent_start_250:], fwd10)
        icr250 = float(ic_r250.mean()) if len(ic_r250) else float("nan")
        icir250 = float(ic_r250.mean() / ic_r250.std()) if len(ic_r250) > 2 and ic_r250.std() > 0 else float("nan")
        ic_r60 = rank_ic_series(sig.loc[recent_start_60:], fwd10)
        icr60 = float(ic_r60.mean()) if len(ic_r60) else float("nan")
        icir60 = float(ic_r60.mean() / ic_r60.std()) if len(ic_r60) > 2 and ic_r60.std() > 0 else float("nan")
        turn = turnover_10d_rank(sig)
        rows.append((fid, icw, icirw, icl, icirl, icr250, icir250, icr60, icir60, turn))
        print(f"  {fid:26s} warm IC={icw:+.4f} ICIR={icirw:+.4f} | live IC={icl:+.4f} ICIR={icirl:+.4f} | "
              f"r250 IC={icr250:+.4f} ICIR={icir250:+.4f} | r60 IC={icr60:+.4f} ICIR={icir60:+.4f} | turn={turn:.2f}")

    # correlation among library factors (last 60d)
    print("\n===== FACTOR PAIRWISE CORR (last 60d, stacked) =====")
    names = list(libsig.keys())
    corr_mat = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            fa = libsig[a].iloc[-60:].stack()
            fb = libsig[b].iloc[-60:].stack()
            j = pd.concat([fa.rename('a'), fb.rename('b')], axis=1).dropna()
            if len(j) > 100:
                corr_mat[f"{a}|{b}"] = float(j['a'].corr(j['b']))
    for k, v in sorted(corr_mat.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:60s} {v:+.3f}")
