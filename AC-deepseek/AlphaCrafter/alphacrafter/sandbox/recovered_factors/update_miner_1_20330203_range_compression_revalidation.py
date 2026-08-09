import json
from pathlib import Path

p = Path('factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs.json')
d = json.loads(p.read_text())
metrics = {
    'selected_horizon_days': 20,
    'daily_paper_ic': 0.062446,
    'daily_paper_icir': 0.185257,
    'hit_ratio': 0.5778,
    'ic_dates': 1561,
    'mean_valid_instruments': 9.39,
    'minimum_valid_instruments': 8,
    'universe_instruments': 15,
    'signal_cell_coverage': 0.3215,
    'signal_cells': '19714/61320',
    'daily_rank_turnover': 0.108516,
    'concentration_median_cross_sectional_iqr': 0.070098,
    'max_abs_library_correlation': 0.459838,
    'closest_library_factor': 'relative_liquidity_stress_20_60obs',
    'library_factors_screened': 30,
    'correlation_evidence_note': 'Unchanged-factor revalidation. Complete library Spearman novelty audit remains valid evidence: maximum absolute rho 0.459838, below 0.500000.',
    'decay': {
        '1d': {'ic': -0.001256, 'icir': -0.003612, 'hit_ratio': 0.5101, 'dates': 1580},
        '5d': {'ic': 0.014422, 'icir': 0.041631, 'hit_ratio': 0.5152, 'dates': 1576},
        '10d': {'ic': 0.054007, 'icir': 0.155864, 'hit_ratio': 0.5582, 'dates': 1571},
        '20d': {'ic': 0.062446, 'icir': 0.185257, 'hit_ratio': 0.5778, 'dates': 1561}
    }
}
d['version'] = '2033-02-03'
d['validation'] = {
    'period': '2026-07-16 through 2033-01-19 completed daily bars; forward-return availability varies by horizon',
    'status': 'EFFECTIVE',
    'metrics': metrics,
    'regime_notes': 'Selected 20d factor passes shared gates. Broad partitions are both positive: 2026-2029 IC 0.059243 / ICIR 0.179506 (849 dates), and 2030-2033-01-19 IC 0.066266 / ICIR 0.191803 (712 dates). Recent performance has recovered versus the prior negative recent-12-month reading, but conditional downside-event coverage remains sparse; retain enhanced monitoring.'
}
d['last_validated'] = '2033-02-03'
d['next_revalidation_due'] = '2033-05-03'
d.setdefault('validation_history', []).append({
    'date': '2033-02-03', 'cutoff': '2033-01-19', 'status': 'EFFECTIVE',
    'period': '2026-07-16 through 2033-01-19 completed daily bars',
    'metrics': metrics,
    'regime_notes': d['validation']['regime_notes'],
    'note': 'Scheduled revalidation recorded durably from completed-bar audit. The unchanged definition clears aggregate gates at 10d and 20d; admission-time complete-library correlation evidence remains below the binding 0.5000 limit.'
})
p.write_text(json.dumps(d, indent=2) + '\n')
print('updated', p)
print('status', d['validation']['status'], '20d IC', metrics['daily_paper_ic'], 'ICIR', metrics['daily_paper_icir'], 'rho', metrics['max_abs_library_correlation'])
