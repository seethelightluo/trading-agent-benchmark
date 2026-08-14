"""miner_3 2034-11-03: library re-validation + candidate screen (fully vectorized beta AND IC).

Vectorization notes:
- roll_beta: rolling cov/var via expanding-mean formulation (pandas rolling, vectorized).
  Series regressor is passed as .values to avoid DataFrame-by-Series cross-product alignment.
- IC: rank along cross-section (axis=1) with pandas rank, then per-date Pearson of ranks.
"""
import pandas as pd
import numpy as np

PANEL = 'scripts/panel_cache_20341103.pkl'

def load_panel():
    with open(PANEL, 'rb') as f:
        return pd.read_pickle(f)

def roll_beta(a, b, w):
    """Vectorized rolling beta of a on b, window w. b: Series or 1-col array."""
    bv = np.asarray(b).ravel() if not isinstance(b, pd.DataFrame) else b
    B = pd.DataFrame({c: bv for c in a.columns}, index=a.index)
    ma = a.rolling(w).mean(); mb = B.rolling(w).mean()
    cov = (a * B).rolling(w).mean() - ma * mb
    var = (B * B).rolling(w).mean() - mb * mb
    beta = cov / var.replace(0, np.nan)
    return beta

def ic_series_vec(fac, fwd, min_valid=8):
    """Spearman IC per date, vectorized via cross-sectional ranks + per-row Pearson."""
    rf = fac.rank(axis=1)
    rr = fwd.rank(axis=1)
    F = rf.values.astype(float)
    R = rr.values.astype(float)
    valid = np.isfinite(F) & np.isfinite(R)
    F[~valid] = np.nan
    R[~valid] = np.nan
    nvalid = valid.sum(axis=1)
    Fm = F - np.nanmean(F, axis=1, keepdims=True)
    Rm = R - np.nanmean(R, axis=1, keepdims=True)
    num = np.nansum(Fm * Rm, axis=1)
    den = np.sqrt(np.nansum(Fm * Fm, axis=1) * np.nansum(Rm * Rm, axis=1))
    ic = np.where(den > 0, num / den, np.nan)
    ic[nvalid < min_valid] = np.nan
    return pd.Series(ic, index=fac.index)

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

def eval_factor(fac, fwd_cache, min_valid=8, lib=None, recent=250):
    out = {}
    for h, fwd in fwd_cache.items():
        s = ic_series_vec(fac, fwd, min_valid=min_valid).dropna()
        if len(s) < 30:
            out[h] = {'n': int(len(s))}
            continue
        icm = s.mean(); icstd = s.std()
        sr = s.tail(recent)
        out[h] = {
            'n': int(len(s)),
            'ic': float(icm),
            'icir': float(icm / icstd) if icstd > 0 else np.nan,
            'hit': float((np.sign(s) == np.sign(icm)).mean()),
            'cov': float(fac.notna().sum().sum() / (fac.shape[0] * fac.shape[1])),
            'ic_recent': float(sr.mean()),
            'icir_recent': float(sr.mean() / sr.std()) if sr.std() > 0 else np.nan,
            'first_date': str(s.index.min().date()),
            'last_date': str(s.index.max().date()),
        }
    rk = fac.rank(axis=1)
    turn = rk.diff().abs().mean().mean() / (rk.shape[1] - 1)
    out['turnover'] = float(turn)
    if lib is not None:
        rho_max, anchor = 0.0, None
        alld = fac.index.intersection(next(iter(lib.values())).index)
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

# precompute forward-return rank cache per horizon (shared by all factors)
fwd_cache = {}
for h in [1, 5, 10]:
    fwd_cache[h] = px.pct_change(h).shift(-h)

lib = make_library_factors_full(panel)
print("=" * 70)
print("LIBRARY RE-VALIDATION (full sample through 2034-11-02)")
print("=" * 70)
for k, v in lib.items():
    res = eval_factor(v, fwd_cache, lib=None)
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
    res = eval_factor(v, fwd_cache, lib=lib)
    print_eval(k, res)
