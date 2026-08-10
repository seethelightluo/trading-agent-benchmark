"""miner_2 exploration: risk / macro-interaction factor families (batch screen).
Candidate families (one screen, several related ideas):
  1. realized skewness (lottery aversion)        - skew(ret,N)
  2. vol term-structure ratio                     - vol(S)/vol(L)
  3. conditional rate beta (US10Y driver)         -beta(asset,US10Y)*dUS10Y
  4. cross-sectional relative momentum            - mom_N - cs_median(mom_N)
  5. low-beta vs equal-weight market              - beta(asset, EW mkt)
  6. max daily return (lottery)                   - max(ret,N)
  7. downside-vol share                           - dd_vol/total_vol
Screen metric: h=10 cross-sectional rank IC / ICIR over 2020-01-01..2026-07-15.
Admission gates |IC|>=0.007, |ICIR|>=0.084. Detailed validation follows for
the best candidates in a separate one-idea script.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_lib import (load_panel, load_macro, per_asset, fwd_returns,
                         rank_ic_series, MIN_ASSETS, FACTOR_LAST, ADMISSION)

# ---------------------------------------------------------------- helpers
def make_skew(n: int):
    def fn(panel, macro):
        return per_asset(lambda s: s.pct_change().rolling(n).skew())(panel, macro)
    return fn

def make_vol_ratio(short_w: int, long_w: int):
    def fn(panel, macro):
        def f(s):
            r = s.pct_change()
            return r.rolling(short_w).std() / r.rolling(long_w).std()
        return per_asset(f)(panel, macro)
    return fn

def make_rate_beta(beta_win: int, mkt_win: int):
    def fn(panel, macro):
        y = load_macro("US10Y")["US10Y"] if False else macro.get("US10Y")
        # US10Y is in the tradable panel; use its level changes as the driver
        us10y = panel["US10Y"].dropna()
        dy = us10y.pct_change()
        def f(s):
            r = s.pct_change()
            z = pd.concat([r.rename("r"), dy.reindex(s.index).rename("y")], axis=1)
            beta = z["r"].rolling(beta_win).cov(z["y"]) / z["y"].rolling(beta_win).var().replace(0, np.nan)
            mv = (us10y / us10y.shift(mkt_win) - 1.0).reindex(s.index)
            return -beta * mv
        return per_asset(f)(panel, macro)
    return fn

def make_rel_mom(n: int, skip: int):
    def fn(panel, macro):
        mom = panel.shift(skip) / panel.shift(n + skip) - 1.0
        med = mom.median(axis=1)
        return mom.sub(med, axis=0)
    return fn

def make_low_beta(win: int):
    def fn(panel, macro):
        rets = panel.pct_change()
        mkt = rets.mean(axis=1)
        def f(s):
            r = s.pct_change()
            z = pd.concat([r.rename("r"), mkt.reindex(s.index).rename("m")], axis=1)
            return z["r"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var().replace(0, np.nan)
        return per_asset(f)(panel, macro)
    return fn

def make_max_ret(n: int):
    def fn(panel, macro):
        return per_asset(lambda s: s.pct_change().rolling(n).max())(panel, macro)
    return fn

def make_downside_vol_ratio(win: int):
    def fn(panel, macro):
        def f(s):
            r = s.pct_change()
            tot = r.rolling(win).std()
            dd = r.clip(upper=0).rolling(win).std()
            return dd / tot
        return per_asset(f)(panel, macro)
    return fn

# ---------------------------------------------------------------- screen
CANDIDATES = {
    "skew_60": make_skew(60),
    "skew_120": make_skew(120),
    "vol_ratio_5_60": make_vol_ratio(5, 60),
    "vol_ratio_20_60": make_vol_ratio(20, 60),
    "rate_beta_cond_60x20": make_rate_beta(60, 20),
    "rel_mom_20d_skip5": make_rel_mom(20, 5),
    "rel_mom_60d_skip5": make_rel_mom(60, 5),
    "low_beta_60": make_low_beta(60),
    "max_ret_20d": make_max_ret(20),
    "max_ret_60d": make_max_ret(60),
    "downside_vol_ratio_20": make_downside_vol_ratio(20),
}

def main():
    panel = load_panel()
    macro = load_macro()
    fwd10 = fwd_returns(panel, 10)
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
            gate = (abs(icm) >= ADMISSION["ic"]) and (abs(icir) >= ADMISSION["icir"])
            print(f"{name:24s} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
                  f"n={len(ic):4d} cov={cov:.3f} {'PASS' if gate else 'fail'}")
            rows.append((name, icm, icir, hit, len(ic), cov))
        except Exception as e:
            print(f"{name:24s} ERROR {type(e).__name__}: {e}")
    print("\nranked by |ICIR|:")
    for r in sorted(rows, key=lambda x: -abs(x[2])):
        print(f"  {r[0]:24s} IC={r[1]:+.4f} ICIR={r[2]:+.4f} hit={r[3]:.3f} n={r[4]} cov={r[5]:.3f}")

if __name__ == "__main__":
    main()
