"""miner2 2029-02-16: Candidate A - Drawdown depth (medium-horizon reversal).
Idea: assets far below their rolling highs tend to mean-revert; distinct horizon (10-120d)
from the 1-5d reversal family already in the library. Positive factor = deeper drawdown.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from miner2_factor_val import (load_panel, library_signals, daily_ic_series, ic_metrics,
                               signal_correlation_matrix, format_metrics, GATE_IC, GATE_ICIR)

panel = load_panel()
close = panel['close']
ret = panel['ret']
fwd1 = ret.shift(-1)

lib = library_signals(panel)
print("library signals recomputed:", len(lib))

for W in (10, 20, 60, 120, 250):
    roll_max = close.rolling(W).max()
    dd = 1.0 - close / roll_max  # positive = deeper drawdown
    ic_s = daily_ic_series(dd, fwd1)
    m = ic_metrics(ic_s, dd, ret, label=f'drawdown_{W}d')
    maxabs, rows = signal_correlation_matrix(dd, lib)
    m['max_abs_library_correlation'] = maxabs
    print("=" * 90)
    print(format_metrics(m))
    print(f"  max_abs_library_corr={maxabs:.3f}")
    top = sorted(rows, key=lambda r: -abs(r[1]))[:4]
    print("  top lib corr: " + " | ".join(f"{n}:{r:.3f}" for n, r in top))
    gate = abs(m['ic']) >= GATE_IC and abs(m['icir']) >= GATE_ICIR
    print(f"  GATE PASS: {gate} (|IC|>={GATE_IC}, |ICIR|>={GATE_ICIR})")
