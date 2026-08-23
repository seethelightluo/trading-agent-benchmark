"""miner_3 sweep AB: market-beta / common-factor exposure factors (2026-11-19).

Fresh dimension vs library (which has VIX/FX betas but no market-portfolio beta).
Candidates:
  1. mkt_beta_60   : rolling beta of asset return vs equal-weight cross-asset market
  2. mkt_beta_20   : short-horizon market beta
  3. mkt_corr_chg_20_60 : corr(asset, market, 20) - corr(asset, market, 60)
  4. idio_mom_60   : idiosyncratic (alpha) momentum - asset ret - beta*market ret, cumulated 60d
  5. mkt_res_drift_20: residual (capm alpha) z-score over 60d window
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from miner3_20260730_harness import ASSETS, evaluate, load_closes, VISIBLE_END

def load_ohlc():
    import pandas as pd, pathlib
    D = pathlib.Path("../persistent/stock_data")
    out = {}
    for a in ASSETS:
        f = D / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        out[a] = df.set_index("date")
    return out

def main():
    closes = load_closes()
    ohlc = load_ohlc()
    print("assets:", len(closes))

    # Equal-weight market return on common calendar
    rets = pd.DataFrame({a: s.pct_change() for a, s in closes.items()})
    mkt_ret = rets.mean(axis=1, skipna=True)
    mkt_ret.name = "mkt"

    cand = {}

    # 1. market beta 60d
    for a in closes:
        r = rets[a]
        beta = (r.rolling(60, min_periods=40).cov(mkt_ret) / mkt_ret.rolling(60, min_periods=40).var().replace(0, np.nan))
        cand.setdefault("mkt_beta_60", {})[a] = beta

    # 2. market beta 20d
    for a in closes:
        r = rets[a]
        beta = (r.rolling(20, min_periods=15).cov(mkt_ret) / mkt_ret.rolling(20, min_periods=15).var().replace(0, np.nan))
        cand.setdefault("mkt_beta_20", {})[a] = beta

    # 3. market corr change 20 vs 60
    for a in closes:
        r = rets[a]
        c20 = r.rolling(20, min_periods=15).corr(mkt_ret)
        c60 = r.rolling(60, min_periods=40).corr(mkt_ret)
        cand.setdefault("mkt_corr_chg_20_60", {})[a] = c20 - c60

    # 4. idiosyncratic momentum 60d: asset - beta20 * market
    for a in closes:
        r = rets[a]
        beta = (r.rolling(20, min_periods=15).cov(mkt_ret) / mkt_ret.rolling(20, min_periods=15).var().replace(0, np.nan))
        idio = (r - beta * mkt_ret).rolling(60, min_periods=40).sum()
        cand.setdefault("idio_mom_60", {})[a] = idio

    # 5. mkt_res_delta_20: residual return z-score vs trailing 120d
    for a in closes:
        r = rets[a]
        beta = (r.rolling(20, min_periods=15).cov(mkt_ret) / mkt_ret.rolling(20, min_periods=15).var().replace(0, np.nan))
        resid = r - beta * mkt_ret
        rd20 = resid.rolling(20, min_periods=15).sum()
        z = (rd20 - rd20.rolling(120, min_periods=60).mean()) / rd20.rolling(120, min_periods=60).std().replace(0, np.nan)
        cand.setdefault("mkt_res_delta_20", {})[a] = z

    for name, vals in cand.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()