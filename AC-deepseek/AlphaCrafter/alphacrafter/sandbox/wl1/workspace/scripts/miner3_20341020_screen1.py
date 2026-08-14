"""miner_3 2034-10-20: library re-validation + candidate screen (vectorized beta)."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

PANEL = 'scripts/panel_cache_20341020.pkl'

def load_panel():
    with open(PANEL, 'rb') as f:
        return pd.read_pickle(f)

def roll_beta(a, b, w):
    """Vectorized rolling beta of a on b, window w."""
    ma = a.rolling(w).mean(); mb = b.rolling(w).mean()
    cov = (a * b).rolling(w).mean() - ma * mb
    var = (b * b).rolling(w).mean() - mb * mb
    beta = cov / var.replace(0, np.nan)
    return beta

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
    vix = panel['macro']['VIX'].reindex(px.index).ffill()
    vix_ret = vix.pct_change()
    betas = roll_beta(ret, vix_ret, 60)
    vix_trend = vix_ret.rolling(20).mean()
    lib['vix_beta_cond_60x20'] = betas * np.sign(vix_trend).values[:, None]
    return lib

def eval_factor(fac, px, horizons=(1, 5, 10), min_valid=8, lib=None, recent=250):
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
            'ic_recent': float(s.tail(recent).mean()),
            'icir_recent': float(s.tail(recent).mean() / s.tail(recent).std()) if s.tail(recent).std() > 0 else np.nan,
            'first_date': str(s.index.min().date()),
            'last_date': str(s.index.max().date()),
        }
    rk = fac.rank(axis=1)
    turn = rk.diff().abs().mean().mean() / (rk.shape[1] - 1)
    out['turnover'] = float(turn)
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
            print(f"  h={h}: n={r['n']} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} cov={r['cov']:.3f} icR250={r['ic_recent']:+.4f} icirR250={r['icir_recent']:+.3f} [{r['first_date']}..{r['last_date']}]")
        else:
            print(f"  h={h}: n={r.get('n','?')} too few")
    print(f"  turnover={res.get('turnover', float('nan')):.3f} rho_max={res.get('rho_max', float('nan')):.3f} anchor={res.get('rho_anchor')}")
    print(f"  GATE(1d |ic|>={gates[0]}, |icir|>={gates[1]}): {'PASS' if ig else 'FAIL'}")
    return ig

panel = load_panel()
px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']

lib = make_library_factors_full(panel)
print("=" * 70)
print("LIBRARY RE-VALIDATION (full sample through 2034-10-19)")
print("=" * 70)
for k, v in lib.items():
    res = eval_factor(v, px, lib=None)
    print_eval(k, res)
print()
print("=" * 70)
print("CANDIDATE FACTORS")
print("=" * 70)

cands = {}
cands['mom_20d_skip5'] = px.shift(5) / px.shift(25) - 1.0
cands['efficiency_20d'] = (px / px.shift(20) - 1.0).abs() / ret.abs().rolling(20).sum()
cands['maxdd_60d'] = -((px / px.rolling(60).max()) - 1.0)
def downside_ratio(s, w=20):
    d = s.clip(upper=0.0)
    return d.rolling(w).std() / s.rolling(w).std()
cands['downside_ratio_20d'] = downside_ratio(ret)
rng = (hi - lo) / px
cands['range_5d'] = rng.rolling(5).mean()
cands['rev_5d_vs'] = -(np.log(px) - np.log(px.shift(5))) / ret.rolling(20).std()
dxy = panel['macro']['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
cands['dxy_beta_60d'] = roll_beta(ret, dxy_ret, 60)
vix = panel['macro']['VIX'].reindex(px.index).ffill()
vix_hi = (vix > 40).astype(float)
rev5 = -(np.log(px) - np.log(px.shift(5)))
cands['rev5_x_vixhi'] = rev5 * (1.0 + 1.0 * vix_hi.values[:, None])
cands['mom_60d_skip5'] = px.shift(5) / px.shift(65) - 1.0
spx_ret = ret['SPX']
cands['rel_20d_vs_spx'] = (px / px.shift(20) - 1.0).subtract(spx_ret.rolling(20).sum(), axis=0)

for k, v in cands.items():
    res = eval_factor(v, px, lib=lib)
    print_eval(k, res)
