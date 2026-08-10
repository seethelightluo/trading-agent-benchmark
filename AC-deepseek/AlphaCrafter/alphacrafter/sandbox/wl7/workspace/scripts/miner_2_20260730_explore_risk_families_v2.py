"""miner_2 exploration (v2): risk / macro-interaction factor families (batch screen).
Fixes vs v1:
  - rel_mom computed per-asset on each asset's own calendar (then reindexed),
    avoiding union-calendar NaN propagation.
  - rate-beta uses US10Y returns computed on US10Y's own calendar, ffill to union.
Adds library-correlation audit to flag redundancy with the 4 effective factors.
Screen metric: h=10 cross-sectional rank IC / ICIR over 2020-01-01..2026-07-15.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import (load_panel, load_macro, per_asset, fwd_returns,
                         rank_ic_series, library_signals, library_corr,
                         MIN_ASSETS, FACTOR_LAST, ADMISSION)

def make_skew(n: int):
    return per_asset(lambda s: s.pct_change().rolling(n).skew())

def make_vol_ratio(short_w: int, long_w: int):
    def f(s):
        r = s.pct_change()
        return r.rolling(short_w).std() / r.rolling(long_w).std()
    return per_asset(f)

def make_rate_beta(beta_win: int, mkt_win: int):
    """-beta(asset_ret, dUS10Y, B) * (US10Y_t/US10Y_{t-M} - 1)."""
    us10y = load_macro().get("US10Y")
    if us10y is None:
        p = load_panel()
        us10y = p["US10Y"].dropna()
    dy = us10y.pct_change()
    dy_u = dy.reindex(pd.date_range(us10y.index.min(), us10y.index.max(), freq="D")).ffill()
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), dy_u.reindex(s.index).rename("y")], axis=1)
        beta = z["r"].rolling(beta_win).cov(z["y"]) / z["y"].rolling(beta_win).var().replace(0, np.nan)
        mv = (us10y / us10y.shift(mkt_win) - 1.0)
        mv = mv.reindex(pd.date_range(mv.index.min(), mv.index.max(), freq="D")).ffill().reindex(s.index)
        return -beta * mv
    return per_asset(f)

def make_rel_mom(n: int, skip: int):
    """per-asset momentum minus cross-sectional median (union reindex)."""
    def f(s):
        return s.shift(skip) / s.shift(n + skip) - 1.0
    def inner(pnl, mcr):
        mom = per_asset(f)(pnl, mcr)
        med = mom.median(axis=1)
        return mom.sub(med, axis=0)
    return inner

def make_low_beta(win: int):
    def f(s):
        r = s.pct_change()
        z = pd.concat([r.rename("r"), mkt.reindex(s.index).rename("m")], axis=1)
        return z["r"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var().replace(0, np.nan)
    return per_asset(f)

def make_max_ret(n: int):
    return per_asset(lambda s: s.pct_change().rolling(n).max())

def make_downside_vol_ratio(win: int):
    def f(s):
        r = s.pct_change()
        tot = r.rolling(win).std()
        dd = r.clip(upper=0).rolling(win).std()
        return dd / tot
    return per_asset(f)

def make_corr_mkt(win: int):
    """rolling correlation of asset returns with EW market return (defensive)."""
    def f(s):
        r = s.pct_change()
        return r.rolling(win).corr(mkt.reindex(s.index))
    return per_asset(f)

panel = load_panel()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)

CANDIDATES = {
    "rel_mom_10d_skip5": make_rel_mom(10, 5),
    "rel_mom_20d_skip5": make_rel_mom(20, 5),
    "rel_mom_60d_skip5": make_rel_mom(60, 5),
    "low_beta_60": make_low_beta(60),
    "low_beta_120": make_low_beta(120),
    "corr_mkt_60": make_corr_mkt(60),
    "max_ret_10d": make_max_ret(10),
    "max_ret_20d": make_max_ret(20),
    "downside_vol_ratio_20": make_downside_vol_ratio(20),
    "downside_vol_ratio_60": make_downside_vol_ratio(60),
    "rate_beta_cond_60x20": make_rate_beta(60, 20),
    "rate_beta_cond_60x60": make_rate_beta(60, 60),
}

def main():
    fwd10 = fwd_returns(panel, 10)
    libs = library_signals(panel)
    print(f"panel {panel.index[0].date()}..{panel.index[-1].date()} assets={panel.shape[1]}")
    rows = []
    for name, fn in CANDIDATES.items():
        try:
            fac = fn(panel, macro).loc[:FACTOR_LAST]
            ic = rank_ic_series(fac, fwd10)
            if len(ic) < 200:
                print(f"{name:24s} insufficient dates: {len(ic)}")
                continue
            icir = float(ic.mean() / ic.std())
            icm = float(ic.mean())
            hit = float((ic > 0).mean())
            cov = float(fac.notna().mean().mean())
            maxc, per = library_corr(fac, panel, libs)
            gate = (abs(icm) >= ADMISSION["ic"]) and (abs(icir) >= ADMISSION["icir"])
            flag = "PASS" if gate else "fail"
            flag += f"  maxLibCorr={maxc:.2f}" if gate else ""
            print(f"{name:24s} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
                  f"n={len(ic):4d} cov={cov:.3f} libcorr={maxc:.3f} {flag}")
            rows.append((name, icm, icir, hit, len(ic), cov, maxc))
        except Exception as e:
            print(f"{name:24s} ERROR {type(e).__name__}: {e}")
    print("\nranked by |ICIR|:")
    for r in sorted(rows, key=lambda x: -abs(x[2])):
        print(f"  {r[0]:24s} IC={r[1]:+.4f} ICIR={r[2]:+.4f} hit={r[3]:.3f} n={r[4]} cov={r[5]:.3f} libcorr={r[6]:.3f}")

if __name__ == "__main__":
    main()
