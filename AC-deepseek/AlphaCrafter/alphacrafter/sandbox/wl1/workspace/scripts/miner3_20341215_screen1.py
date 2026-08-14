"""miner_3 2034-12-15: library re-validation + candidate screen on fresh panel through 2034-12-14.

Fully vectorized (rolling ops, per-date rank IC). Candidates this cycle focus on:
- candlestick structure (clv, wicks, body, gap) - orthogonal short-horizon signal
- vol-adjusted / horizon reversal variants
- trend persistence (autocorrelation, efficiency)
- vol/risk regime (vol ratio, downside vol, skew)
- macro beta (DXY/VIX, shorter windows)
"""
import pandas as pd
import numpy as np

PANEL = 'scripts/panel_cache_20341215.pkl'

def load_panel():
    with open(PANEL, 'rb') as f:
        return pd.read_pickle(f)

def roll_beta(a, b, w):
    bv = np.asarray(b).ravel() if not isinstance(b, pd.DataFrame) else b
    B = pd.DataFrame({c: bv for c in a.columns}, index=a.index)
    ma = a.rolling(w).mean(); mb = B.rolling(w).mean()
    cov = (a * B).rolling(w).mean() - ma * mb
    var = (B * B).rolling(w).mean() - mb * mb
    beta = cov / var.replace(0, np.nan)
    return beta

def ic_series_vec(fac, fwd, min_valid=8):
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

fwd_cache = {}
for h in [1, 5, 10]:
    fwd_cache[h] = px.pct_change(h).shift(-h)

lib = make_library_factors_full(panel)
print("=" * 70)
print("LIBRARY RE-VALIDATION (full sample through 2034-12-14)")
print("=" * 70)
for k, v in lib.items():
    res = eval_factor(v, fwd_cache, lib=None)
    print_eval(k, res)
print()
print("=" * 70)
print("CANDIDATE FACTORS (2034-12-15 cycle)")
print("=" * 70)

cands = {}
rng = (hi - lo) / px
oc_range = (hi - lo).replace(0, np.nan)
# --- candlestick structure ---
cands['clv_1d'] = ((px - lo) - (hi - px)) / oc_range          # close location value
cands['upper_wick_1d'] = (hi - np.maximum(op, px)) / oc_range  # upper wick ratio
cands['lower_wick_1d'] = (np.minimum(op, px) - lo) / oc_range  # lower wick ratio
cands['body_1d'] = (px - op) / oc_range                        # body direction
cands['gap_1d'] = op / px.shift(1) - 1.0                       # overnight gap
# --- reversal variants ---
cands['rev_4d'] = -(np.log(px) - np.log(px.shift(4)))
cands['rev_10d'] = -(np.log(px) - np.log(px.shift(10)))
cands['rev_1d_vs20'] = -(np.log(px) - np.log(px.shift(1))) / ret.rolling(20).std()
cands['rev_5d_vs20'] = -(np.log(px) - np.log(px.shift(5))) / ret.rolling(20).std()
# --- trend persistence ---
def ac_ret(s, w=10):
    out = pd.DataFrame(index=s.index, columns=s.columns, dtype=float)
    for c in s.columns:
        x = s[c]
        out[c] = x.rolling(w).apply(lambda z: np.corrcoef(z[:-1], z[1:])[0, 1] if len(z) >= 4 and np.std(z[:-1]) > 0 and np.std(z[1:]) > 0 else np.nan, raw=True)
    return out
cands['ac_10d'] = ac_ret(ret, 10)
cands['efficiency_60d'] = (px / px.shift(60) - 1.0).abs() / ret.abs().rolling(60).sum()
cands['zscore_20d'] = (px - px.rolling(20).mean()) / ret.rolling(20).std()
cands['ma20_dist'] = px / px.rolling(20).mean() - 1.0
# --- vol / risk regime ---
cands['vol_ratio_5_60'] = ret.rolling(5).std() / ret.rolling(60).std()
cands['downside_vol_20d'] = ret.clip(upper=0).rolling(20).std()
cands['skew_20d'] = ret.rolling(20).skew()
cands['maxdd_20d'] = -((px / px.rolling(20).max()) - 1.0)
# --- macro beta ---
dxy = panel['macro']['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
cands['dxy_beta_20d'] = roll_beta(ret, dxy_ret, 20)
vix = panel['macro']['VIX'].reindex(px.index).ffill()
vix_ret = vix.pct_change()
cands['vix_beta_20d'] = roll_beta(ret, vix_ret, 20)

for k, v in cands.items():
    res = eval_factor(v, fwd_cache, lib=lib)
    print_eval(k, res)
