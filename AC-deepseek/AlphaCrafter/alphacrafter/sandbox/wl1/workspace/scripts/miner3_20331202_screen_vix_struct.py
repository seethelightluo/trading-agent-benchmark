"""miner3 2033-12-02: screen family A - VIX-structure / relative-fear factors.
Motivation: VIX at ~44.6 (elevated crisis zone), high cross-sectional dispersion.
Hypothesis: assets whose realized vol is LOW relative to market fear (VIX) or with
low VIX-beta are defensive and outperform in the current elevated-fear regime.
Novel vs library (vix_beta_cond_60x20 uses beta x sign(VIX 20d trend); these use
level-based conditioning / unconditional beta / vol-vs-VIX spread).
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

panel = load_panel()
px = panel['close']; ret = panel['ret']
lib = make_library_factors_full(panel)
vix = panel['macro']['VIX'].reindex(px.index).ffill()
vix_ret = vix.pct_change()
ann = np.sqrt(252)

def beta_to(vals, window=60):
    """rolling beta of asset returns vs macro series (aligned index)."""
    betas = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
    for i in range(window, len(ret)):
        a = ret.iloc[i-window:i]; b = vals.iloc[i-window:i]
        m = a.notna() & b.notna()
        if int(m.sum().sum()) < 10:
            continue
        aa = a[m]; bb = b[m]
        cov = (aa * bb).mean() - aa.mean() * bb.mean()
        var = bb.var()
        if var > 0:
            betas.iloc[i] = cov / var
    return betas

def factor_vix_beta_level_60():
    """unconditional 60d beta to VIX returns (positive = risk-asset)"""
    return beta_to(vix_ret, 60)

def factor_vol_spread_vix_20():
    """asset ann realized vol 20d minus VIX level (pct). Low = defensive vs fear."""
    return ret.rolling(20).std() * ann - vix

def factor_vol_ratio_vix_20():
    """asset 20d realized vol / VIX (relative calm)"""
    return ret.rolling(20).std() * ann / vix

def factor_vix_beta_cond_level_60():
    """60d VIX beta x (VIX > 40) - crisis-conditioned risk exposure"""
    return beta_to(vix_ret, 60) * (vix > 40).astype(float).values[:, None]

cands = {
    'vix_beta_level_60': factor_vix_beta_level_60,
    'vol_spread_vix_20': factor_vol_spread_vix_20,
    'vol_ratio_vix_20': factor_vol_ratio_vix_20,
    'vix_beta_cond_level_60': factor_vix_beta_cond_level_60,
}

for name, fn in cands.items():
    try:
        fac = fn()
        res = eval_factor(fac, px, lib=lib)
        print_eval(name, res)
    except Exception as e:
        print(f"ERROR {name}: {e}")
    print()
