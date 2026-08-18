"""miner_3 2028-04-06: full validation for flip_mom_20x10 (lead candidate of flip-momentum family).
Computes leave-one-asset-out, full-metrics, and builds signal artifact panel for persistence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats, spearman_panel_rho

ASOF = '2028-04-05'
H = 10
px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index

def vseries(s):
    return s.dropna()

def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(s, p, kw=20, ks=10):
    m20 = retk(p, kw)
    m10 = retk(p, ks)
    return (m20 * np.sign(m10)).reindex(INDEX)

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
f = build(lambda s, p: flip_mom(s, p, 20, 10)).replace([np.inf, -np.inf], np.nan)

print("--- leave-one-asset-out IC (flip_mom_20x10) ---")
lo = {}
for s in WATCH:
    fsub = f.drop(columns=[s])
    icd = cross_sectional_ic(fsub, fwd10.drop(columns=[s]))
    st = ic_stats(icd)
    lo[s] = [round(st['ic'], 4), round(st['icir'], 4), int(st['n_dates'])]
    print(f"  drop {s:10s}: ic={st['ic']:.4f} icir={st['icir']:.4f} n={st['n_dates']}")

# rank stability / turnover by horizon
ranks = f.rank(axis=1)
for hh in [5, 10, 20]:
    to = ranks.diff(hh).abs().mean().mean() / (len(WATCH) - 1)
    print(f"turnover_rank_{hh}d = {to:.4f}")

# factor panel tail for signal artifact persistence (weekly snapshots, last 520 rows)
sig = f.tail(520).copy()
sig['date'] = sig.index
sig = sig[['date'] + WATCH]
sig.to_csv('scripts/_flip_mom_20x10_signal_20280405.csv', index=False)
print("signal artifact rows:", len(sig), "cols:", sig.shape[1])
print(sig.tail(2).to_string())
