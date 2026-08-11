"""miner_1 cycle 2026-08-13: validate overnight_gap_20.

overnight_gap_20 = rolling 20d mean of overnight returns: open_t/close_{t-1} - 1.

Motivation: in a cross-asset universe (equity indices, crypto, FX-sensitive
commodities), overnight gaps embed macro/news information flow that is distinct
from intraday drift captured by close-to-close momentum factors. Assets that
persistently gap in one direction may carry information drift (or, if news-
driven, mean-revert). Direction determined empirically.

Gate: |IC| >= 0.007, |ICIR| >= 0.084 on 15-asset universe, max-abs library
rho < 0.5 (computed from real signal artifacts).
"""
import json, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import load_panel, closes_panel, forward_returns, ic_series, summary_metrics, regime_split
from miner3_lib import decode_artifact, build_artifact

VIS = '2026-08-12'
H = 10
close = closes_panel(VIS)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}", flush=True)

# ---- signal: overnight gap mean over 20d ----
src = load_panel(visible_through=VIS, source='stock')
opens = pd.DataFrame({s: df.set_index('date')['open'].astype(float) for s, df in src.items()}).sort_index()
closes2 = pd.DataFrame({s: df.set_index('date')['close'].astype(float) for s, df in src.items()}).sort_index()
opens = opens.reindex(columns=close.columns)
closes2 = closes2.reindex(columns=close.columns)
gap = opens / closes2.shift(1) - 1.0
sig = gap.rolling(20, min_periods=8).mean()
print(f"gap coverage: {sig.notna().mean().mean():.3f} asset-days, ge8 dates: {sig.dropna(thresh=8).shape[0]}/{len(sig)}", flush=True)

fr = forward_returns(close, H)
ic_s = ic_series(sig, fr, min_valid=8)
m = summary_metrics(ic_s, sig, fr, close, h=H)
if m is None:
    print("insufficient IC dates", flush=True)
    sys.exit(1)
m['regime'] = regime_split(ic_s)

# library max-abs Spearman rho from real artifacts (scan factors/, incl. kurt_20)
best = 0.0
rhos = {}
for fn in sorted(os.listdir('factors')):
    if not fn.endswith('.json') or fn == 'factor_ensemble.json' or fn.endswith('.bak'):
        continue
    d = json.load(open(f'factors/{fn}'))
    art = d.get('validation', {}).get('signal_artifact')
    if not art:
        continue
    libp = decode_artifact(art).reindex(close.index)
    common = sig.index.intersection(libp.index)
    a = sig.loc[common].stack()
    b = libp.loc[common].stack()
    mm = a.notna() & b.notna()
    if mm.sum() >= 200:
        r = float(a[mm].rank().corr(b[mm].rank()))
        if np.isfinite(r):
            rhos[fn] = round(r, 3)
            best = max(best, abs(r))
m['library_spearman_rho'] = rhos
m['max_abs_library_correlation'] = round(best, 3)
print("library spearman rho:", json.dumps(rhos, indent=1), flush=True)

gate = bool(abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084 and best < 0.5)
print(f"=== overnight_gap_20: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
      f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
      f"max_rho_lib={best} GATE={gate}", flush=True)
print("  decay:", m['decay_ic_by_horizon'], flush=True)
print("  regimes:", m['regime'], flush=True)
print("RESULT:", "PASS" if gate else "FAIL", flush=True)
