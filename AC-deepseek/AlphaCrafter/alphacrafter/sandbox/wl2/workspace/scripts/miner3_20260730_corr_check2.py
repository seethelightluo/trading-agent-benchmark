"""miner_3 2026-07-30: correlation check round 2 - include mkt_beta_20 & max_ret_20."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, VISIBLE

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


def f_mom10(s):
    return s.shift(5) / s.shift(15) - 1.0


def f_mom120(s):
    return s.shift(5) / s.shift(125) - 1.0


def f_vov(s):
    return s.pct_change().rolling(20).std().rolling(60).std()


market = ret.mean(axis=1)
def f_mktbeta(s, w=20):
    r = s.pct_change()
    z = pd.concat([r.rename("a"), market.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(w).cov(z["m"])
    var = z["m"].rolling(w).var()
    return (cov / var).reindex(r.index)


signals = {
    "cand_price_zscore_60": per_asset(lambda s: f_zscore(s, 60)),
    "cand_upday_ratio_20": per_asset(lambda s: f_upday(s, 20)),
    "cand_max_ret_20": per_asset(lambda s: f_maxret(s, 20)),
    "cand_mkt_beta_20": per_asset(lambda s: f_mktbeta(s, 20)),
    "lib_mom_10d_skip5": per_asset(f_mom10),
    "lib_mom_120d_skip5": per_asset(f_mom120),
    "lib_vol_of_vol20x60": per_asset(f_vov),
}
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
print(f"{'':28s}" + "".join(f"{n[5:20]:>16s}" for n in names))
for i, a in enumerate(names):
    row = []
    for j, b in enumerate(names):
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        c = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = c
        row.append(c)
    print(f"{a:28s}" + "".join(f"{c:>16.3f}" for c in row))

print("\nmax |rho| of each candidate vs library factors (lib_*):")
for a in ["cand_price_zscore_60", "cand_upday_ratio_20", "cand_max_ret_20", "cand_mkt_beta_20"]:
    libs = [rho.loc[a, b] for b in names if b.startswith("lib_")]
    print(f"  {a:24s} max_abs={max(abs(c) for c in libs):.3f}  all={[round(c,3) for c in libs]}")

print("\nmax |rho| among candidates:")
cands = [a for a in names if a.startswith("cand_")]
for i, a in enumerate(cands):
    for b in cands[i + 1:]:
        print(f"  {a[5:]:22s} vs {b[5:]:22s} rho={rho.loc[a, b]:+.3f}")
