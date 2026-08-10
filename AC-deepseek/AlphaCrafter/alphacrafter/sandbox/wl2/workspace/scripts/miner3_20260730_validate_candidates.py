"""miner_3 2026-07-30: validate 4 new candidates + re-validate 4 library factors."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, forward_returns, spearman_ic, mean_rank_turnover, ADMISSION_HORIZON

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()


def per_asset(func):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def f_zscore(s, w=60):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def f_upday(s, w=20):
    return (s.pct_change() > 0).rolling(w).mean()


def f_maxret(s, w=20):
    return s.pct_change().rolling(w).max()


market = ret.mean(axis=1)
def f_mktbeta(s, w=20):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), market.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(w).cov(z["m"])
    var = z["m"].rolling(w).var()
    return (cov / var).reindex(r.index)


def f_mom10(s):
    return s.shift(5) / s.shift(15) - 1.0


def f_mom120(s):
    return s.shift(5) / s.shift(125) - 1.0


def f_vov(s):
    return s.pct_change().rolling(20).std().rolling(60).std()


cands = {
    "cand_price_zscore_60": per_asset(lambda s: f_zscore(s, 60)),
    "cand_upday_ratio_20": per_asset(lambda s: f_upday(s, 20)),
    "cand_max_ret_20": per_asset(lambda s: f_maxret(s, 20)),
    "cand_mkt_beta_20": per_asset(lambda s: f_mktbeta(s, 20)),
    "lib_mom_10d_skip5": per_asset(f_mom10),
    "lib_mom_120d_skip5": per_asset(f_mom120),
    "lib_vol_of_vol20x60": per_asset(f_vov),
}

fwd = forward_returns(prices, ADMISSION_HORIZON)
for name, fdf in cands.items():
    ic_series = spearman_ic(fdf, fwd)
    ic = float(ic_series.mean())
    icir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
    turn = mean_rank_turnover(fdf)
    print(f"{name:24s} n={len(ic_series):5d} ic={ic:+.4f} icir={icir:+.4f} "
          f"|ic|={abs(ic):.4f} |icir|={abs(icir):.4f} pass_gate={'YES' if abs(ic)>=0.007 and abs(icir)>=0.084 else 'no'} "
          f"turn={turn:.3f}")
