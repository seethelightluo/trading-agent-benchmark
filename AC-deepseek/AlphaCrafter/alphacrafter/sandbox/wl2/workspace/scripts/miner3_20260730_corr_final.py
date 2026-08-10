"""miner_3 2026-07-30: full pairwise rho incl. vix_beta + self-contained expression test."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel, load_obs

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()
vix = load_obs('VIX')


def per_asset(func, extra=None):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s, extra).reindex(panel.index) if extra is not None else func(s).reindex(panel.index)
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


def f_vixbeta_cond(s, v):
    """-beta(asset_ret, vix_ret, 60) * vix_20d_mom"""
    r = s.pct_change().rename('a')
    vr = v.pct_change().rename('v')
    z = pd.concat([r, vr], axis=1).dropna()
    beta = (z['a'].rolling(60).cov(z['v']) / z['v'].rolling(60).var())
    vix_mom = v / v.shift(20) - 1.0
    return (-beta * vix_mom.reindex(z.index)).reindex(s.index)


signals = {
    "cand_price_zscore_60": per_asset(lambda s: f_zscore(s, 60)),
    "cand_upday_ratio_20": per_asset(lambda s: f_upday(s, 20)),
    "cand_mkt_beta_20": per_asset(lambda s: f_mktbeta(s, 20)),
    "lib_mom_10d_skip5": per_asset(f_mom10),
    "lib_mom_120d_skip5": per_asset(f_mom120),
    "lib_vol_of_vol20x60": per_asset(f_vov),
    "lib_vix_beta_cond_60x20": per_asset(f_vixbeta_cond, extra=vix),
}
names = list(signals.keys())
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        c = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = c

print("pairwise |rho| matrix (lower triangle):")
for i, a in enumerate(names):
    row = "".join(f"{rho.loc[a,b]:>8.3f}" if j <= i else "        " for j, b in enumerate(names))
    print(f"{a[5:]:22s}{row}")

print("\nmax |rho| vs library (lib_*):")
for a in [n for n in names if n.startswith('cand_')]:
    libs = [rho.loc[a, b] for b in names if b.startswith('lib_')]
    print(f"  {a:24s} max_abs={max(abs(c) for c in libs):.3f} all={[round(c,3) for c in libs]}")

print("\nmax |rho| among library factors:")
for i, a in enumerate([n for n in names if n.startswith('lib_')]):
    for b in [n for n in names if n.startswith('lib_')]:
        if a < b:
            print(f"  {a[4:]:22s} vs {b[4:]:22s} rho={rho.loc[a,b]:+.3f}")

# --- self-contained expression test under the gate namespace ---
print("\n--- self-contained expression eval test (gate namespace: close, pct_change, pd, np) ---")
env = {'pd': pd, 'np': np, 'close': panel, 'pct_change': ret}
exprs = {
    'lib_mom_10d_skip5': 'close.shift(5) / close.shift(15) - 1.0',
    'lib_mom_120d_skip5': 'close.shift(5) / close.shift(125) - 1.0',
    'lib_vol_of_vol20x60': 'pct_change().rolling(20).std().rolling(60).std()',
    'cand_price_zscore_60': '(close - close.rolling(60).mean()) / close.rolling(60).std()',
    'cand_upday_ratio_20': '(pct_change() > 0).rolling(20).mean()',
    'cand_mkt_beta_20': 'pct_change().rolling(20).cov(pct_change().mean(axis=1)) / pct_change().mean(axis=1).rolling(20).var()',
}
for name, exp in exprs.items():
    try:
        sig = eval(exp, {'__builtins__': {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape and sig.notna().sum().sum() > 0
        rho_vs_ref = None
        if ok and name in signals:
            both = pd.concat([sig.stack().rename('x'), signals[name].stack().rename('y')], axis=1).dropna()
            rho_vs_ref = round(float(both['x'].corr(both['y'])), 4) if len(both) > 100 else None
        print(f"  {name:24s} eval={'OK' if ok else 'FAIL'} shape={sig.shape if ok else '-'} rho_vs_reference={rho_vs_ref}")
    except Exception as e:
        print(f"  {name:24s} eval=FAIL err={type(e).__name__}: {str(e)[:80]}")
