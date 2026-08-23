"""miner_3 (2026-08-21): Sweep W - fresh orthogonal dimensions round 2.

Prior sweeps produced many gate-PASS candidates but most carried high library
correlation (cs_beta_20 0.558 vs beta_VIX; cs_mom_rank20 0.604 vs rng_pos;
up_down_20 0.801; range_pos_z 0.840). Here probe genuinely distinct centers:

  - amihud_illiq_20 : Amihud illiquidity |ret|/volume averaged over 20d (liquidity)
  - down_side_vol60 : downside semi-deviation scaled by total vol (asym risk)
  - coskew_20       : covariance(asset ret, market ret^2)/... (co-moment skewness)
  - vol_term_5_60   : 5d realized vol / 60d realized vol (vol term structure)
  - pos_vol_ratio60 : upside-vol / downside-vol asymmetry (distinct from up/down capture)
  - dxy_beta_60     : rolling 60d beta vs DXY (fresh macro center; beta_DXY was evicted
                      for high corr with beta_VIX, but this beta-on-catching may differ)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro


def rolling_beta(a, m, w, minp=12):
    df = pd.concat([a.rename("a"), m.rename("m")], axis=1)
    out = []
    for i in range(len(df)):
        if i < w - 1:
            out.append(np.nan); continue
        sub = df.iloc[i-w+1:i+1]
        mm = sub["m"].to_numpy(); aa = sub["a"].to_numpy()
        fm = np.isfinite(mm) & np.isfinite(aa)
        if fm.sum() < minp or np.nanstd(mm[fm]) == 0:
            out.append(np.nan); continue
        out.append(np.cov(aa[fm], mm[fm])[0, 1] / np.var(mm[fm]))
    return pd.Series(out, index=df.index)


def main():
    closes = load_closes()
    macro = load_macro()
    ohlc = {}
    for a in ASSETS:
        import pathlib
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        ohlc[a] = df.set_index("date")

    ret = {a: closes[a].pct_change() for a in closes}
    mkt = pd.DataFrame(ret).mean(axis=1)  # equal-weight CS mean
    retf = pd.DataFrame(ret)

    cand = {}

    # Amihud illiquidity: |ret| / dollar volume, averaged 20d
    cand["amihud_illiq_20"] = {
        a: (ret[a].abs() / ohlc[a]["volume"].astype(float).replace(0, np.nan)).rolling(20).mean()
        for a in closes
    }

    # downside semi-deviation / total vol over 60d
    ds = {}
    tv = {}
    for a in closes:
        m = ret[a].rolling(60).mean()
        d = (ret[a] - m).clip(upper=0)
        ssd = np.sqrt((d ** 2).rolling(60).mean())
        tv[a] = ret[a].rolling(60).std()
        ds[a] = ssd / tv[a].replace(0, np.nan)
    cand["down_side_vol60"] = ds

    # coskewness: E[(r_a - mu_a)(r_m - mu_m)^2] / (std_a * var_m)
    csk = {}
    mu = mkt.rolling(20).mean()
    mdev = (mkt - mu) ** 2
    for a in closes:
        mu_a = ret[a].rolling(20).mean()
        num = ((ret[a] - mu_a) * mdev).rolling(20).mean()
        csk[a] = num / (ret[a].rolling(20).std().replace(0, np.nan) * mkt.rolling(20).var().replace(0, np.nan))
    cand["coskew_20"] = csk

    # vol term structure 5d/60d
    vt = {}
    for a in closes:
        vt[a] = ret[a].rolling(5).std() / ret[a].rolling(60).std().replace(0, np.nan)
    cand["vol_term_5_60"] = vt

    # upside-vol / downside-vol ratio over 60d (signed asymmetry)
    uv = {}
    for a in closes:
        up = (ret[a].clip(lower=0) - ret[a].rolling(60).mean())
        dn = (ret[a] - ret[a].rolling(60).mean()).clip(upper=0)
        uvol = np.sqrt((up ** 2).rolling(60).mean())
        dvol = np.sqrt((dn ** 2).rolling(60).mean())
        uv[a] = (uvol - dvol) / (uvol + dvol).replace(0, np.nan)
    cand["updown_vol_ratio60"] = uv

    # DXY beta (fresh macro beta center on level changes)
    dxy = macro["DXY"]
    dxy_ret = dxy.pct_change().reindex(closes["SPX"].index)
    cand["dxy_beta_60"] = {a: rolling_beta(ret[a], dxy_ret, 60) for a in closes}

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()
