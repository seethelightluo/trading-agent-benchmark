"""miner_3 2026-07-30: persist rsi_14 (round-6 admit).

Passes admission gates on shared 15-asset universe:
  IC=+0.0497 (>= 0.007), ICIR=+0.1529 (>= 0.084) at 10d horizon,
  max_abs_library_correlation=0.465 vs vol_adj_mom_20_60 (< 0.5).
Writes factors/rsi_14.json + factors/rsi_14_signal.npy, then verifies reload.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, canonical_grid, factor_to_panel,
                           validate_factor, max_library_correlation,
                           persist_factor, load_artifact_matrix, Path, WATCHLIST)

t0 = time.time()
prices = load_prices(days=3000)
grid = canonical_grid(prices)


def rsi_14(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-r).clip(lower=0).rolling(14, min_periods=7).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).replace([np.inf, -np.inf], np.nan)


panel = factor_to_panel(rsi_14, prices)
print(f"rsi_14 panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()} "
      f"({time.time()-t0:.1f}s)")

metrics = validate_factor('rsi_14', panel, prices)
print("validate:", {k: v for k, v in metrics.items() if k != 'decay_ic_by_horizon'})
print("decay:", json.dumps(metrics['decay_ic_by_horizon'], default=str))

# library correlation vs all 12 effective artifact factors
lib = {}
for jp in sorted(Path('factors').glob('*.json')):
    if jp.name == 'factor_ensemble.json':
        continue
    payload = json.loads(jp.read_text(encoding='utf-8'))
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = load_artifact_matrix(str(jp))
    if art is None or art.shape[0] != len(grid) or art.shape[1] != 15:
        continue
    lib[payload['factor_id']] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
rho, best = max_library_correlation(panel, lib)
metrics['max_abs_library_correlation'] = rho
metrics['max_corr_library_id'] = best
print(f"max library rho = {rho:.4f} vs {best}")

assert abs(metrics['ic']) >= 0.007, "IC gate failed"
assert abs(metrics['icir']) >= 0.084, "ICIR gate failed"
assert rho < 0.5, f"correlation gate failed: {rho}"

path, arr = persist_factor(
    factor_id='rsi_14',
    factor_name='RSI-14 Relative Strength Index (cross-asset)',
    expression='100 - 100/(1 + RS), RS = mean(max(ret,0),14)/mean(max(-ret,0),14)',
    description='Classic Wilder RSI-14 on daily closes: ratio of average up-move to '
                'average down-move over 14 sessions, mapped to 0..100. High values mark '
                'assets with strong recent up-pressure (overbought), low values mark '
                'down-pressure (oversold). In this cross-asset universe the 10d forward '
                'IC is positive: higher RSI (stronger recent momentum) assets tended to '
                'earn higher forward 10d returns, consistent with a momentum/trend regime '
                'over 2020-2026. Distinct from vol_adj_mom_20_60 (max pairwise rho 0.465).',
    dependencies=['close'],
    parameters={'window': 14, 'min_periods': 7, 'admission_horizon': 10,
                'signal_floor': 0.0, 'signal_cap': 100.0},
    expected_direction='positive (IC>0)',
    panel=panel,
    metrics=metrics,
    tags=['momentum', 'oscillator', 'trend', 'cross-asset'],
    grid=grid,
    regime_notes='Validated 2020-01-01..2026-07-15 across equity, commodity, crypto, '
                 'and rate assets; strongest at h=10 (IC 0.0497) with monotone decay '
                 'building from h=2; positive hit ratio 0.548; survives correlation gate '
                 'vs 12-factor library (max rho 0.465 vs vol_adj_mom_20_60).',
)
print(f"persisted: {path}")
print(f"artifact: factors/rsi_14_signal.npy shape {arr.shape}")

# ---- verification: read back and re-validate JSON ----
payload = json.loads(Path(path).read_text(encoding='utf-8'))
assert payload['factor_id'] == 'rsi_14'
assert payload['validation']['status'] == 'EFFECTIVE'
assert payload['validation']['metrics']['ic'] == metrics['ic']
assert payload['validation']['metrics']['icir'] == metrics['icir']
assert abs(payload['validation']['metrics']['ic']) >= 0.007
assert abs(payload['validation']['metrics']['icir']) >= 0.084
assert payload['validation']['metrics'].get('max_abs_library_correlation') == rho
art2 = load_artifact_matrix(path)
assert art2 is not None and art2.shape == (len(grid), 15), f"artifact reload failed {art2.shape if art2 is not None else None}"
assert np.allclose(art2, arr, equal_nan=True)
print("VERIFY OK: JSON reloadable, status EFFECTIVE, gates hold, artifact recoverable "
      f"({art2.shape[0]}x{art2.shape[1]}), rho={rho:.4f}")
print(f"TOTAL {time.time()-t0:.1f}s")
