"""miner3 shared evaluation library (2033-12-02).
Loads the panel cache and evaluates a candidate factor DataFrame vs forward returns.
Also reconstructs the existing effective library factors for correlation audit.
FIX: VIX macro series is reindexed/ffilled onto the asset calendar before beta
computation (previous version sliced by position on different-length indices).
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

PANEL = 'scripts/panel_cache_20331202.pkl'

def load_panel():
    with open(PANEL, 'rb') as f:
        return pd.read_pickle(f)

def make_library_factors_full(panel):
    px = panel['close']; ret = panel['ret']
    hi = panel['high']; lo = panel['low']; op = panel['open']
    lib = {}
    for n in [1, 2, 3, 5]:
        lib[f'rev_{n}d'] = -(np.log(px) - np.log(px.shift(n)))
        rmax = px.rolling(n).max(); rmin = px.rolling(n).min()
        lib[f'nclv_{n}d'] = -(px - rmin) / (rmax - rmin)
    lib['id_rev_1d'] = -(px / px.shift(1) - 1.0)
    lib['nbody_1d'] = -((px - op) / (hi - lo))
    lib['rev_1d_vs'] = -(np.log(px) - np.log(px.shift(1))) / ret.rolling(20).std()
    lib['mom_120d_skip5'] = px.shift(5) / px.shift(125) - 1.0
    lib['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
    # vix_beta_cond_60x20 (aligned: VIX reindexed to asset calendar)
    vix = panel['macro']['VIX'].reindex(px.index).ffill()
    vix_ret = vix.pct_change()
    betas = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    a_ret = ret
    for i in range(60, len(a_ret)):
        a = a_ret.iloc[i-60:i]; b = vix_ret.iloc[i-60:i]
        m = a.notna() & b.notna()
        if int(m.sum().sum()) < 10:
            continue
        aa = a[m]; bb = b[m]
        cov = (aa * bb).mean() - aa.mean() * bb.mean()
        var = bb.var()
        if var > 0:
            betas.iloc[i] = cov / var
    vix_trend = vix_ret.rolling(20).mean()
    lib['vix_beta_cond_60x20'] = betas * np.sign(vix_trend).values[:, None]
    return lib

def eval_factor(fac, px, horizons=(1, 5, 10), min_valid=8, lib=None):
    """fac: DataFrame (dates x assets). Returns dict of metrics per horizon."""
    out = {}
    ret = px.pct_change()
    for h in horizons:
        fwd = px.pct_change(h).shift(-h)
        ics = {}
        for dt in fac.index:
            f = fac.loc[dt]; r = fwd.loc[dt]
            m = f.notna() & r.notna()
            if int(m.sum()) >= min_valid:
                rho, _ = spearmanr(f[m], r[m])
                ics[dt] = rho
        s = pd.Series(ics)
        if len(s) < 30:
            out[h] = {'n': len(s)}
            continue
        icm = s.mean(); icstd = s.std()
        out[h] = {
            'n': int(len(s)),
            'ic': float(icm),
            'icir': float(icm / icstd) if icstd > 0 else np.nan,
            'hit': float((np.sign(s) == np.sign(icm)).mean()),
            'cov': float(fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])),
            'first_date': str(s.index.min().date()),
            'last_date': str(s.index.max().date()),
        }
    # turnover: mean abs daily change of cross-sectional rank
    rk = fac.rank(axis=1)
    turn = rk.diff().abs().mean().mean() / (rk.shape[1] - 1)
    out['turnover'] = float(turn)
    # library correlation (on overlapping dates, cross-sectional flattened)
    if lib is not None:
        rho_max, anchor = 0.0, None
        alld = fac.index.intersection(lib['rev_1d'].index)
        for k, v in lib.items():
            vv = v.loc[alld]
            mm = fac.loc[alld].notna() & vv.notna()
            if int(mm.sum().sum()) < 200:
                continue
            a = fac.loc[alld][mm].values.flatten()
            b = vv[mm].values.flatten()
            rho = np.corrcoef(a, b)[0, 1]
            if abs(rho) > abs(rho_max):
                rho_max, anchor = rho, k
        out['rho_max'] = float(rho_max)
        out['rho_anchor'] = anchor
    return out

def print_eval(name, res, gates=(0.0070, 0.0840)):
    h1 = res.get(1, {})
    ig = abs(h1.get('ic', 0)) >= gates[0] and abs(h1.get('icir', 0)) >= gates[1]
    print(f"--- {name} ---")
    for h in [1, 5, 10]:
        r = res.get(h, {})
        if 'ic' in r:
            print(f"  h={h}: n={r['n']} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} cov={r['cov']:.3f} [{r['first_date']}..{r['last_date']}]")
        else:
            print(f"  h={h}: n={r.get('n','?')} too few")
    print(f"  turnover={res.get('turnover', float('nan')):.3f} rho_max={res.get('rho_max', float('nan')):.3f} anchor={res.get('rho_anchor')}")
    print(f"  GATE(1d |ic|>={gates[0]}, |icir|>={gates[1]}): {'PASS' if ig else 'FAIL'}")
    return ig
