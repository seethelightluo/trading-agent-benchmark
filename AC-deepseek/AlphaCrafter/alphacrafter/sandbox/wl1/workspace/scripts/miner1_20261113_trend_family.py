"""miner_1 cycle 2026-11-13: explore trend/momentum family candidates.

Context: trader feedback (memory 20260910) - reversal-family drag for 3 blocks;
requested trend/momentum guards. Explore volatility-scaled momentum,
distance-from-high, MA-slope trend, and vol-conditioned momentum.

Validates on full history 2020-01-02..2026-11-12 (visible-through) plus a
recent window 2025-01-01..2026-11-12 for drift check.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_close_panel():
    frames = {}
    for sym in TRADABLE:
        df = get_stock_daily_data(symbol=sym, days=2500)
        s = df.set_index("date")["close"].rename(sym)
        frames[sym] = s
    return pd.concat(frames, axis=1).sort_index()


def daily_rank_ic(factor, fwd_ret, min_obs=8):
    dates, ics = [], []
    for dt in factor.index:
        f, r = factor.loc[dt], fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_obs:
            continue
        ic = f[m].rank().corr(r[m].rank())
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def summarize(ic_series):
    ics = ic_series.dropna()
    n = len(ics)
    if n == 0:
        return None
    mean_ic = ics.mean()
    std_ic = ics.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())
    return {"ic": round(float(mean_ic), 5), "icir": round(float(icir), 5),
            "ic_hit_ratio": round(hit, 4), "n_ic_dates": int(n)}


def turnover_10d(factor, horizon_days=10):
    rank = factor.rank(axis=1)
    return float(rank.diff(horizon_days).abs().mean().mean())


def build_library_factors(close):
    lib = {
        "rev_1d": -close.pct_change(),
        "rev_2d": -(close / close.shift(2) - 1),
        "rev_3d": -(close / close.shift(3) - 1),
        "mom_10d_skip5": close.shift(5) / close.shift(15) - 1,
        "mom_120d_skip5": close.shift(5) / close.shift(125) - 1,
        "trend_20d": close / close.rolling(20).mean() - 1,
        "trend_60d": close / close.rolling(60).mean() - 1,
    }
    return lib


def library_corr(factor, lib, min_obs=8):
    out = {}
    for name, lf in lib.items():
        corrs = []
        for dt in factor.index:
            if dt not in lf.index:
                continue
            f, g = factor.loc[dt], lf.loc[dt]
            m = f.notna() & g.notna()
            if m.sum() < min_obs:
                continue
            c = f[m].rank().corr(g[m].rank())
            if np.isfinite(c):
                corrs.append(c)
        out[name] = round(float(np.mean(corrs)), 4) if corrs else None
    return out


def validate(factor_panel, close, label, win_start=None, win_end=None):
    fp = factor_panel
    cl = close
    if win_start is not None:
        fp = fp[fp.index >= win_start]
        cl = cl[cl.index >= win_start]
    if win_end is not None:
        fp = fp[fp.index <= win_end]
        cl = cl[cl.index <= win_end]
    results = {}
    for h in (1, 2, 3, 5, 10, 20):
        fwd = cl.shift(-h) / cl - 1.0
        ic_series = daily_rank_ic(fp, fwd)
        results[h] = summarize(ic_series)
    admitted = None
    for h in (1, 2, 3, 5, 10):
        r = results[h]
        if r and abs(r["ic"]) >= GATE_IC and abs(r["icir"]) >= GATE_ICIR:
            if admitted is None or abs(r["icir"]) > abs(results[admitted]["icir"]):
                admitted = h
    cov_assets = float(fp.notna().mean().mean())
    cov_dates_ge8 = float(fp.notna().sum(axis=1).ge(8).mean())
    to10 = turnover_10d(fp)
    lib = build_library_factors(close)
    lc = library_corr(fp, lib)
    max_abs_lc = max((abs(v) for v in lc.values() if v is not None), default=0.0)
    print(f"\n=== {label}  [{fp.index[0].date()} .. {fp.index[-1].date()}] n_dates={len(fp)} n_assets={fp.shape[1]}")
    for h, r in results.items():
        if r:
            print(f"  h={h:2d}  IC={r['ic']:+.5f}  ICIR={r['icir']:+.5f}  hit={r['ic_hit_ratio']:.3f}  n={r['n_ic_dates']}")
    print(f"  >> ADMISSION h={admitted}" + (f" IC={results[admitted]['ic']:+.5f} ICIR={results[admitted]['icir']:+.5f}" if admitted else " (none)"))
    print(f"  coverage_asset_days={cov_assets:.3f} cov_dates_ge8={cov_dates_ge8:.3f} turnover_10d_rank={to10:.3f}")
    print(f"  library_corr={lc}  max_abs={max_abs_lc:.4f}")
    return {"label": label, "admitted": admitted, "results": results, "max_abs_lc": max_abs_lc}


def main():
    close = load_close_panel()
    ret = close.pct_change()
    print(f"close panel: {close.shape}  [{close.index[0].date()} .. {close.index[-1].date()}]")

    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()

    candidates = {
        # volatility-scaled 60d momentum (risk-adjusted trend)
        "mom60_vol20": close / close.shift(60) - 1,
        "mom60_vol20_s": (close / close.shift(60) - 1) / vol20,
        "mom120_vol20_s": (close.shift(5) / close.shift(125) - 1) / vol20,
        # distance from 52w high (trend strength / recovery)
        "dth_252": close / close.rolling(252).max() - 1,
        "dth_120": close / close.rolling(120).max() - 1,
        # MA slope golden-cross style
        "ma_slope_20x60": (close.rolling(20).mean() / close.rolling(60).mean() - 1),
        # vol-conditioned momentum: momentum only strong when vol regime low
        "mom60_lowvol": (close / close.shift(60) - 1) * (vol20 <= vol20.rolling(120).median()),
        "vol_ratio_20x60": vol20 / vol60,
    }

    print("\n########## FULL WINDOW VALIDATION ##########")
    for name, fac in candidates.items():
        validate(fac, close, name)

    print("\n########## RECENT WINDOW 2025-01-01..2026-11-12 ##########")
    for name, fac in candidates.items():
        validate(fac, close, name + "_recent", win_start="2025-01-01")


if __name__ == "__main__":
    main()
