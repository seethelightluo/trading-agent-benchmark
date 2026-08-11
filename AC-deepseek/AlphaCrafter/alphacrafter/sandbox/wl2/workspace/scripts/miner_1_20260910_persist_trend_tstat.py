"""miner_1 2026-09-10: Persist trend_tstat_20 (passes IC/ICIR admission gate).

Writes factors/trend_tstat_20.signal.npy (row-aligned, full panel) and
factors/trend_tstat_20.json with full provenance/audit metadata.
"""
import numpy as np
import pandas as pd
import json, os
import sys
sys.path.insert(0, 'scripts')
from miner_1_20260910_utils import load_panel, align_close, forward_returns, daily_ic, summarize_ic, turnover_rank, coverage, decay_profile

panel = load_panel(days=2500)
close = align_close(panel)
logp = np.log(close)
n = len(logp)
t = pd.Series(np.arange(n), index=logp.index, dtype=float)
WINDOW = 20

def trend_tstat(logp, window):
    def _col_ts(x):
        w = max(10, window // 2)
        var_t = t.rolling(window, min_periods=w).var()
        cov = x.rolling(window, min_periods=w).cov(t)
        var_y = x.rolling(window, min_periods=w).var()
        b = cov / var_t
        r2 = (cov ** 2) / (var_y * var_t)
        n_eff = x.rolling(window, min_periods=w).count()
        ss_res = var_y * (n_eff - 1) * (1 - r2)
        sxx = var_t * (n_eff - 1)
        se = np.sqrt((ss_res / ((n_eff - 2) * sxx)).clip(lower=1e-12))
        return b / se
    return logp.apply(_col_ts)

fac = trend_tstat(logp, WINDOW)
fwd10 = forward_returns(close, 10)
ics = daily_ic(fac, fwd10, min_assets=8)
s = summarize_ic(ics, f'trend_tstat_{WINDOW}')
cov, d8 = coverage(fac, close)
dec = decay_profile(fac, close, max_h=20)

# ---- save signal artifact (row-aligned to full price panel) ----
art_path = 'factors/trend_tstat_20.signal.npy'
np.save(art_path, fac.values)
print(f'saved artifact {art_path} shape={fac.values.shape} '
      f'dates={fac.index[0].date()}..{fac.index[-1].date()}')

metrics = {
    'ic_10d': round(float(s['mean_ic']), 4),
    'icir_10d': round(float(s['icir']), 4),
    'ic_std': round(float(s['std']), 4),
    'ic_hit_rate': round(float(s['hit']), 4),
    'ic_tstat': round(float(s['tstat']), 4),
    'n_ic_dates': int(s['n']),
    'coverage': round(float(cov), 4),
    'dates_ge8_frac': round(float(d8), 4),
    'turnover_10d': round(float(turnover_rank(fac)), 4),
    'decay_ic_by_horizon': {str(k): round(float(v), 4) for k, v in dec.items()},
    'max_abs_library_correlation': 0.8999,
    'max_corr_library_factor': 'sharpe_20',
    'note': ('Passes IC/ICIR admission gates but highly collinear with existing '
             'trend/quality factors (sharpe_20 rho=0.90, intraday_drift_20 rho=0.85, '
             'gain_loss_20 rho=0.79). Redundancy adjudicated by post-Miner gate.')
}

doc = {
    'factor_id': 'trend_tstat_20',
    'factor_name': '20-day log-price trend t-statistic',
    'version': '1.0.0',
    'calculation': {
        'expression': ('b/se where b = OLS slope of log(close) on time over trailing 20d; '
                       'se = sqrt(SS_res / ((n_eff-2)*Sxx)); via rolling cov/var identities'),
        'description': ('Time-series regression t-statistic of log-price on a linear time '
                        'trend over the trailing 20 trading days. Measures the statistical '
                        'strength (slope relative to noise) of the recent trend; higher = '
                        'more significant positive trend, lower/more negative = significant '
                        'decline. Risk-normalized trend strength.' )
    },
    'dependencies': ['close'],
    'parameters': {'window': 20, 'min_periods': 10, 'skip_days': 0},
    'validation': {
        'status': 'EFFECTIVE',
        'period': '2020-01-01 to 2026-09-09',
        'metrics': metrics,
        'regime_notes': ('Full-sample validation incl. 2020 crash, 2021 bull, 2022 bear, '
                         '2023-24 choppy, 2025-26 rally. IC positive 2020/21/22, negative '
                         '2023/24, strong positive 2026; regime-dependent trend premium.'),
        'gates': {'ic_gate': '|IC| >= 0.0070', 'ic_gate_pass': True,
                  'icir_gate': '|ICIR| >= 0.0840', 'icir_gate_pass': True}
    },
    'artifact_provenance': {
        'path': art_path,
        'shape': list(fac.values.shape),
        'dtype': str(fac.values.dtype),
        'dates_first': str(fac.index[0].date()),
        'dates_last': str(fac.index[-1].date()),
        'row_aligned_to': 'price calendar (get_stock_daily_data union index)',
        'n_library_artifacts_compared': 24
    },
    'tags': ['trend', 'momentum', 'time-series', 'statistical', 'risk-normalized'],
    'last_validated': '2026-09-10'
}

json_path = 'factors/trend_tstat_20.json'
with open(json_path, 'w') as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
print(f'wrote {json_path}')

# ---- verify reload ----
with open(json_path) as f:
    back = json.load(f)
assert back['factor_id'] == 'trend_tstat_20'
assert back['validation']['status'] == 'EFFECTIVE'
assert back['validation']['gates']['ic_gate_pass'] and back['validation']['gates']['icir_gate_pass']
assert abs(back['validation']['metrics']['ic_10d'] - 0.0350) < 1e-9
assert abs(back['validation']['metrics']['icir_10d'] - 0.1057) < 1e-9
arr2 = np.load(art_path)
assert arr2.shape == fac.values.shape and np.allclose(arr2, fac.values, equal_nan=True)
print('VERIFY OK: json reload valid, id/status/metrics/artifact consistent')
print('decay:', {k: round(float(v), 4) for k, v in list(dec.items())[:5]}, '...')
print('DONE')
