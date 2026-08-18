"""miner_3 2028-03-23: robustness confirm for flip_mom family (trend-consistent momentum).
Slow spearman IC via common helper + parameter variants + per-asset coverage + asset-ID decomposition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats

ASOF = '2028-03-22'
H = 10
px = load_prices(ASOF)
INDEX = px.index

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

# variants
variants = {
    'flip_mom_20x5': lambda s, p: flip_mom(s, p, 20, 5),
    'flip_mom_40x5': lambda s, p: flip_mom(s, p, 40, 5),
    'flip_mom_20x10': lambda s, p: flip_mom(s, p, 20, 10),
    'flip_mom_40x10': lambda s, p: flip_mom(s, p, 40, 10),
    'flip_mom_60x10': lambda s, p: flip_mom(s, p, 60, 10),
    'flip_mom_120x10': lambda s, p: flip_mom(s, p, 120, 10),
}

for name, fn in variants.items():
    f = build(fn).replace([np.inf, -np.inf], np.nan)
    icd = cross_sectional_ic(f, fwd10)
    st = ic_stats(icd)
    # recent 252d
    icr = icd[icd.index >= icd.index[-1] - pd.Timedelta(days=252)]
    st_r = ic_stats(icr)
    # coverage per asset (last 252d)
    cov_last = f.tail(252).notna().mean()
    print(f"== {name} == ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avgN={st.get('avg_n', np.nan):.1f}")
    print(f"   recent252 ic={st_r['ic']:.4f} icir={st_r['icir']:.4f} n={st_r['n_dates']}")
    print(f"   cov252={cov_last.mean():.3f} min={cov_last.min():.3f} assets_cov<0.5: {list(cov_last[cov_last<0.5].index)}")

# detailed asset-wise contribution for main candidate: IC on subsets excluding each asset
f_main = build(variants['flip_mom_20x5']).replace([np.inf, -np.inf], np.nan)
print("\n--- leave-one-asset-out IC (flip_mom_20x5) ---")
for s in WATCH:
    fsub = f_main.drop(columns=[s])
    icd = cross_sectional_ic(fsub, fwd10.drop(columns=[s]))
    st = ic_stats(icd)
    print(f"  drop {s:10s}: ic={st['ic']:.4f} icir={st['icir']:.4f} n={st['n_dates']}")

# last-60d factor values to sanity-check direction
print("\n--- latest factor cross-section (rank) ---")
last = f_main.iloc[-1]
print(last.dropna().sort_values().to_string())