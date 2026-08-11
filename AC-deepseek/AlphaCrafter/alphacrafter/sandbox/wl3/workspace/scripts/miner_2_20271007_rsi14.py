"""miner_2 2027-10-07: RSI(14) mean-reversion candidate.
Factor = (50 - RSI_14)/50. High RSI (overbought) -> negative signal -> expected positive IC on fwd 10d.
One idea per script. Uses shared factor_common validation + full-library correlation audit."""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, evaluate_candidate, canonical_grid
from miner2_common import load_effective_artifacts, max_library_correlation

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3000)
grid = canonical_grid(prices)
print(f"prices={len(prices)} grid={len(grid)} dates {grid.min().date()}..{grid.max().date()} ({time.time()-t0:.1f}s)", flush=True)

def rsi_14(df, s):
    c = df['close']
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0/14, adjust=False).mean()
    al = loss.ewm(alpha=1.0/14, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (50.0 - rsi) / 50.0

metrics, panel = evaluate_candidate('rsi_14', rsi_14, prices, print_out=True)
if metrics is None:
    sys.exit(0)

# full-library correlation audit (all effective artifacts)
artifacts = load_effective_artifacts()
rho, fid, details = max_library_correlation(panel, artifacts, grid)
metrics['max_abs_library_correlation'] = rho
metrics['max_corr_library_id'] = fid
print(f"LIB CORR (n={len(artifacts)} artifacts): max|rho|={rho:.4f} vs {fid}")
top = sorted(details.items(), key=lambda kv: -abs(kv[1]))[:5]
print("top corr:", [(k, round(v, 3)) for k, v in top])

ok = abs(metrics['ic']) >= 0.007 and abs(metrics['icir']) >= 0.084
print(f"ADMISSION FINAL: |IC|={abs(metrics['ic']):.4f} |ICIR|={abs(metrics['icir']):.4f} -> {'PASS' if ok else 'FAIL'}")
json.dump({'metrics': {k: v for k, v in metrics.items() if k != 'decay_ic_by_horizon'},
           'decay': metrics['decay_ic_by_horizon'], 'rho': rho, 'max_corr_id': fid,
           'pass': ok}, open('scripts/miner_2_20271007_rsi14.json', 'w'), indent=1, default=str)
print(f"done {time.time()-t0:.1f}s")
