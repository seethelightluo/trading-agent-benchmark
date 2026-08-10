"""miner_3 2026-07-30: pairwise correlation analysis for passing candidates vs
existing library factor signals (recomputed from expressions) to ensure the
persisted set is mutually distinct (pairwise |rho| < 0.5 preferred)."""
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


def f_mom(s, w=252, skip=21):
    return s.shift(skip) / s.shift(skip + w) - 1.0


def f_upday(s, w=20):
    return (s.pct_change() > 0).rolling(w).mean()


def f_vol_mom(s, w1=20, w2=60):
    r = s.pct_change()
    mom = s / s.shift(w1) - 1.0
    return np.sign(mom) * r.rolling(w2).std()


def f_mom10(s):
    return s.shift(5) / s.shift(15) - 1.0


def f_mom120(s):
    return s.shift(5) / s.shift(125) - 1.0


def f_vov(s):
    r = s.pct_change()
    return r.rolling(20).std().rolling(60).std()


signals = {
    "cand_price_zscore_60": per_asset(lambda s: f_zscore(s, 60)),
    "cand_mom_252_skip21": per_asset(lambda s: f_mom(s, 252, 21)),
    "cand_upday_ratio_20": per_asset(lambda s: f_upday(s, 20)),
    "cand_vol_mom_signed_20x60": per_asset(lambda s: f_vol_mom(s, 20, 60)),
    "lib_mom_10d_skip5": per_asset(f_mom10),
    "lib_mom_120d_skip5": per_asset(f_mom120),
    "lib_vol_of_vol20x60": per_asset(f_vov),
}
# vix_beta_cond_60x20 uses VIX macro; approximate with BTC as risk proxy is not valid -
# instead recompute true VIX beta conditional factor
vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix[vix["date"] <= pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float)
vr = vix.pct_change()
vix_ret = vr.reindex(panel.index)


def rolling_beta(r, m, win=60):
    z = pd.concat([r.rename("a"), m.rename("m")], axis=1).dropna()
    cov = z["a"].rolling(win).cov(z["m"])
    var = z["m"].rolling(win).var()
    return (cov / var).reindex(r.index)


def f_vix_beta(s):
    r = s.pct_change()
    beta = rolling_beta(r, vix_ret, 60)
    cond = vix_ret.rolling(20).mean() * -1.0  # conditional on VIX falling over 20d (risk-on)
    return beta * cond
signals["lib_vix_beta_cond_60x20"] = per_asset(f_vix_beta)

names = list(signals.keys())
print("pairwise pooled Pearson |rho| on raw factor values (union reindex):")
print(f"{'':28s}" + "".join(f"{n[5:22]:>18s}" for n in names))
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    row = []
    for j, b in enumerate(names):
        x = signals[a].stack()
        y = signals[b].stack()
        both = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        c = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = c
        row.append(c)
    print(f"{a:28s}" + "".join(f"{c:>18.3f}" for c in row))

print("\nmax |rho| of each candidate vs library factors (lib_*):")
for a in ["cand_price_zscore_60", "cand_mom_252_skip21", "cand_upday_ratio_20", "cand_vol_mom_signed_20x60"]:
    libs = [rho.loc[a, b] for b in names if b.startswith("lib_")]
    print(f"  {a:28s} max_abs={max(abs(c) for c in libs):.3f}  all={[round(c,3) for c in libs]}")

print("\nmax |rho| among candidates (pairwise):")
cands = [a for a in names if a.startswith("cand_")]
for i, a in enumerate(cands):
    for b in cands[i + 1:]:
        print(f"  {a[5:]:24s} vs {b[5:]:24s} rho={rho.loc[a, b]:+.3f}")
