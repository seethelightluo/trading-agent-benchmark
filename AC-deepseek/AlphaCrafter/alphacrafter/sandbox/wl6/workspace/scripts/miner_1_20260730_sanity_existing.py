"""Sanity check: re-validate existing library factors with miner_1 machinery."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
from miner_1_20260730_validation_lib import (load_close_panel,
    forward_returns, rank_ic_series, summarize, library_factor_signals)

panel = load_close_panel()
print('panel shape:', panel.shape, 'range:', panel.index.min().date(), '->', panel.index.max().date())
fwd = forward_returns(panel)
lib = library_factor_signals(panel)

for name, sig in lib.items():
    ic_s, n_s = rank_ic_series(sig, fwd[10])
    s = summarize(ic_s, n_s, name, fwd=fwd, factor_df=sig)
    print(f"\n{name}: IC={s['ic']:.4f} ICIR={s['icir']:.4f} hit={s['hit']:.3f} "
          f"n_dates={s['n_dates']} mean_n={s['mean_n_assets']:.1f} "
          f"cov={s['coverage_asset_days']:.3f} turn={s.get('turnover_rank_abs', np.nan):.2f}")
    print("  decay:", {k: round(v, 4) for k, v in s['decay_ic'].items()})
