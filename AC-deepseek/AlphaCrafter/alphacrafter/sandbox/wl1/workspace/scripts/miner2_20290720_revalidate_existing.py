"""miner2 2029-07-20: revalidate existing effective library factors on data through 2029-07-19."""
import sys, json
import pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from miner2_factor_val_fast import (load_panel, library_signals, daily_ic_series, ic_metrics,
                                    signal_correlation_matrix, format_metrics, GATE_IC, GATE_ICIR)

panel = load_panel('scripts/panel_cache_20290720.pkl')
close = panel['close']
ret = panel['ret']
fwd1 = ret.shift(-1)
lnc = np.log(close)

# guard against macro lookahead: truncate macro to close index max
panel['macro'] = panel['macro'][panel['macro'].index <= close.index.max()]

CUT = '2021-01-01'
mask_idx = close.index >= CUT
close_m = close[mask_idx]
ret_m = ret[mask_idx]
fwd1_m = fwd1[mask_idx]

lib = library_signals(panel)

h, l, o = panel['high'], panel['low'], panel['open']
factors = {}
for nd in (1, 2, 3, 5):
    factors[f'rev_{nd}d'] = -(lnc - lnc.shift(nd))
    rng = h.rolling(nd).max() - l.rolling(nd).min()
    factors[f'nclv_{nd}d'] = -(close - l.rolling(nd).min()) / rng
factors['nbody_1d'] = -(close - o) / (h - l)
factors['id_rev_1d'] = -(close / o - 1.0)
factors['rev_1d_vs'] = -(lnc - lnc.shift(1)) / ret.rolling(20).std()
factors['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
factors['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
macro = panel['macro']
vix = macro['VIX']
vix_ret = vix.pct_change()
beta_vix = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
factors['vix_beta_cond_60x20'] = -beta_vix * (vix / vix.shift(20) - 1.0)

results = {}
for name, F in factors.items():
    Fm = F[mask_idx]
    ic_s = daily_ic_series(Fm, fwd1_m)
    m = ic_metrics(ic_s, Fm, fwd1_m, label=name)
    maxabs, rows = signal_correlation_matrix(Fm, lib)
    m['max_abs_library_correlation'] = maxabs
    m['gate'] = bool(abs(m['ic']) >= GATE_IC and abs(m['icir']) >= GATE_ICIR)
    r12 = m.get('recent_12m')
    if r12:
        m['recent_gate'] = bool(abs(r12['ic']) >= GATE_IC and abs(r12['icir']) >= GATE_ICIR)
    else:
        m['recent_gate'] = False
    results[name] = m
    print("=" * 90)
    print(format_metrics(m))
    print(f"  max_abs_library_corr={maxabs:.3f} | GATE(full) PASS: {m['gate']} | GATE(12m) PASS: {m.get('recent_gate')}")

with open('scripts/miner2_20290720_reval_existing.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved scripts/miner2_20290720_reval_existing.json")
