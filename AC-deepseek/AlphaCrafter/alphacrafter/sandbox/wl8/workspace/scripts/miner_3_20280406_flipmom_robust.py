"""miner_3 2028-04-06: robustness confirm for flip_mom family (trend-consistent momentum).
ASOF = visible_through (2028-04-05). H=10. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
Checks: variant grid, recent-252d timeliness, per-asset coverage, leave-one-asset-out,
library correlation vs active library (usdcny_beta_60) + fallback panels, decay, regime split.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats, spearman_panel_rho

ASOF = '2028-04-05'
H = 10
GATE_IC, GATE_ICIR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index
print(f"price panel: {px.shape[0]} dates x {px.shape[1]} assets; macro {macro.shape}; asof={ASOF}")

def vseries(s):
    return s.dropna()

def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(s, p, kw=20, ks=5):
    m20 = retk(p, kw)
    m5 = retk(p, ks)
    return (m20 * np.sign(m5)).reindex(INDEX)

def build(fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

fwd10 = fwd_panel(H)

variants = {
    'flip_mom_20x5':  lambda s, p: flip_mom(s, p, 20, 5),
    'flip_mom_40x5':  lambda s, p: flip_mom(s, p, 40, 5),
    'flip_mom_20x10': lambda s, p: flip_mom(s, p, 20, 10),
    'flip_mom_40x10': lambda s, p: flip_mom(s, p, 40, 10),
    'flip_mom_60x10': lambda s, p: flip_mom(s, p, 60, 10),
    'flip_mom_120x10':lambda s, p: flip_mom(s, p, 120, 10),
}

# --- library panels for rho: active usdcny_beta_60 + fallback ensemble ---
def rbeta(y, x, w, cond=None, minp=None):
    vy, vx = vseries(y), vseries(x)
    df = pd.concat([vy.rename('y'), vx.rename('x')], axis=1, sort=True).dropna()
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        c = pd.Series(True, index=df.index)
    ym, xm = df['y'].where(c), df['x'].where(c)
    if minp is None:
        minp = max(6, int(w * 0.4))
    cov = ym.rolling(w, min_periods=minp).cov(xm)
    var = xm.rolling(w, min_periods=minp).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(INDEX)

def ret1(s):
    return vseries(s).pct_change().reindex(INDEX)

vix = macro['VIX']
usdcny = macro['USDCNY']
us10 = px['US10Y']

def build_lib(fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

lib_panels = {
    'usdcny_beta_60': build_lib(lambda s, p: rbeta(p, usdcny, 60)),
    'mom_10d_skip5': build_lib(lambda s, p: retk(p, 15) - retk(p, 5)),
    'vix_beta_cond_60x20': build_lib(lambda s, p: rbeta(p, vix, 60, cond=(vix.pct_change().rolling(20).sum() > 0).reindex(p.index))),
    'yield_beta_cond_60x20': build_lib(lambda s, p: rbeta(p, us10, 60, cond=(us10.pct_change().rolling(20).sum() > 0).reindex(p.index))),
}

results = {}
for name, fn in variants.items():
    f = build(fn).replace([np.inf, -np.inf], np.nan)
    icd = cross_sectional_ic(f, fwd10)
    st = ic_stats(icd)
    icr = icd[icd.index >= icd.index[-1] - pd.Timedelta(days=252)] if len(icd) else icd
    st_r = ic_stats(icr)
    cov_last = f.tail(252).notna().mean()
    # turnover (10d rank change)
    ranks = f.rank(axis=1)
    to10 = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    # decay
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        icd_h = cross_sectional_ic(f, fwd_panel(hh))
        decay[hh] = round(float(icd_h['ic'].mean()) if len(icd_h) else np.nan, 4)
    # regime
    reg = {}
    for lab, m in [('2020-2021', icd.index < pd.Timestamp('2022-01-01')),
                   ('2022-2023', (icd.index >= pd.Timestamp('2022-01-01')) & (icd.index < pd.Timestamp('2024-01-01'))),
                   ('2024+', icd.index >= pd.Timestamp('2024-01-01'))]:
        sub = icd[m]
        if len(sub):
            ss = ic_stats(sub)
            reg[lab] = [round(float(ss['ic']), 4), round(float(ss['icir']), 4), int(ss['n_dates'])]
    # library rho
    rhos = {}
    for lname, lp in lib_panels.items():
        rhos[lname] = round(spearman_panel_rho(f, lp.reindex(f.index)), 4)
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    gate = (abs(st['ic']) >= GATE_IC) and (abs(st['icir']) >= GATE_ICIR)
    results[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'], 'avg_n': st.get('avg_n', np.nan),
                     'recent252_ic': st_r['ic'], 'recent252_icir': st_r['icir'], 'recent252_n': st_r['n_dates'],
                     'cov252': float(cov_last.mean()), 'cov252_min': float(cov_last.min()),
                     'turnover_10d': to10, 'decay': decay, 'regime': reg, 'rho_lib': rhos, 'max_lib_rho': maxrho,
                     'gate': bool(gate)}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avgN={st.get('avg_n', np.nan):.1f}")
    print(f"   recent252 ic={st_r['ic']:.4f} icir={st_r['icir']:.4f} n={st_r['n_dates']} | cov252={cov_last.mean():.3f} min={cov_last.min():.3f} to10={to10:.3f} maxrho={maxrho:.3f}")
    print(f"   decay={decay} regime={reg}")

# --- leave-one-asset-out for main candidate ---
f_main = build(variants['flip_mom_20x5']).replace([np.inf, -np.inf], np.nan)
print("\n--- leave-one-asset-out IC (flip_mom_20x5) ---")
lo = {}
for s in WATCH:
    fsub = f_main.drop(columns=[s])
    icd = cross_sectional_ic(fsub, fwd10.drop(columns=[s]))
    st = ic_stats(icd)
    lo[s] = [round(st['ic'], 4), round(st['icir'], 4), int(st['n_dates'])]
    print(f"  drop {s:10s}: ic={st['ic']:.4f} icir={st['icir']:.4f} n={st['n_dates']}")
results['flip_mom_20x5']['leave_one_out'] = lo

print("\n--- latest factor cross-section (rank) ---")
last = f_main.iloc[-1]
print(last.dropna().sort_values().to_string())

json.dump(results, open('scripts/_miner3_20280406_flipmom_robust.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner3_20280406_flipmom_robust.json")
