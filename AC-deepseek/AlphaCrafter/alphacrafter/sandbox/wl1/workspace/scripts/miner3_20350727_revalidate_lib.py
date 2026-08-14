"""miner_3 2035-07-27: re-validate the 14 effective library factors on the fresh panel (through 2035-07-26).
Checks full-history and recent-250d IC/ICIR for drift; gates: |ic|>=0.0070, |icir|>=0.0840 (1d horizon)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

with open('scripts/panel_cache_20350726.pkl', 'rb') as f:
    panel = pd.read_pickle(f)
px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']
vix = panel['macro']['VIX'].reindex(px.index).ffill()

def build_lib():
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

def eval_ics(fac, px, h=1, min_valid=8, window=None):
    fwd = px.pct_change(h).shift(-h)
    fac2 = fac if window is None else fac.iloc[-window:]
    fwd2 = fwd.loc[fac2.index]
    ics = {}
    for dt in fac2.index:
        f = fac2.loc[dt]; r = fwd2.loc[dt]
        m = f.notna() & r.notna()
        if int(m.sum()) >= min_valid:
            rho, _ = spearmanr(f[m], r[m])
            ics[dt] = rho
    s = pd.Series(ics)
    if len(s) < 30:
        return {'n': len(s)}
    icm = s.mean(); icstd = s.std()
    return {'n': int(len(s)), 'ic': float(icm), 'icir': float(icm/icstd) if icstd > 0 else np.nan,
            'hit': float((np.sign(s) == np.sign(icm)).mean()),
            'first': str(s.index.min().date()), 'last': str(s.index.max().date())}

lib = build_lib()
print(f"{'factor':<22} {'h1 full IC/ICIR':>18} {'h1 last250 IC/ICIR':>20} {'h5 last250':>14} {'h10 last250':>14} {'full n':>8}")
for k, v in lib.items():
    fh = eval_ics(v, px, 1, window=None)
    r250_1 = eval_ics(v, px, 1, window=250)
    r250_5 = eval_ics(v, px, 5, window=250)
    r250_10 = eval_ics(v, px, 10, window=250)
    fh_s = f"{fh.get('ic', float('nan')):+.4f}/{fh.get('icir', float('nan')):+.3f}" if 'ic' in fh else "n<30"
    r1_s = f"{r250_1.get('ic', float('nan')):+.4f}/{r250_1.get('icir', float('nan')):+.3f}" if 'ic' in r250_1 else "n<30"
    r5_s = f"{r250_5.get('ic', float('nan')):+.4f}/{r250_5.get('icir', float('nan')):+.3f}" if 'ic' in r250_5 else "n<30"
    r10_s = f"{r250_10.get('ic', float('nan')):+.4f}/{r250_10.get('icir', float('nan')):+.3f}" if 'ic' in r250_10 else "n<30"
    print(f"{k:<22} {fh_s:>18} {r1_s:>20} {r5_s:>14} {r10_s:>14} {fh.get('n','?'):>8}")

# regime snapshot
print("\n--- regime snapshot @2035-07-26 ---")
print("VIX:", round(float(vix.iloc[-1]), 2), "| VIX 20d ago:", round(float(vix.iloc[-21]), 2))
print("20d x-sect dispersion:", round(float(ret.tail(20).std(axis=1).mean()), 5))
print("breadth above MA20:", int((px.tail(20) > px.rolling(20).mean()).iloc[-1].sum()), "/15")
print("breadth above MA60:", int((px > px.rolling(60).mean()).iloc[-1].sum()), "/15")
print("20d cum eqw ret:", round(float((1 + ret.tail(20).mean(axis=1)).prod() - 1), 4))
