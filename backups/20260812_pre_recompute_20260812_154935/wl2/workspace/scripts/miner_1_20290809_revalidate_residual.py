import json
from pathlib import Path
p=Path('factors/miner_1_20290726_agreement_gated_residual_reversal_5d.json')
d=json.loads(p.read_text())
d['version']='1.1'
d['validation']['period']='2020-01-01 through 2029-08-08'
d['validation']['metrics'].update({
 'daily_ic':0.101570,
 'daily_icir':0.266250,
 'ic_hit_ratio':0.6000,
 'valid_dates':250,
 'average_instruments':8.96,
 'coverage':0.5973333333,
 'turnover':None,
 'max_abs_library_correlation':None
})
d['validation']['regime_notes']='Revalidated on the 15-instrument tradable cross-asset universe through 2029-08-08. Full sample uses 250 dates and average 8.96 instruments per date; 2028+ IC/ICIR 0.084413/0.212964 over 83 dates and 2029+ 0.149655/0.436047 over 43 dates. Gated cross-sections remain small, so uncertainty is interpreted conservatively.'
d['validation']['signal_artifact']='scripts/miner_1_20290726_agreement_residual_signal.csv'
d['last_validated']='2029-08-09T00:00:00Z'
d['benchmark_admission']['selected_metrics'].update({'ic':0.101570,'icir':0.266250,'quality':0.027038025})
d['benchmark_admission']['admitted_at']='2029-08-09T00:00:00Z'
p.write_text(json.dumps(d,indent=2)+'\n')
# reload verification
x=json.loads(p.read_text())
assert x['factor_id']=='miner_1_20290726_agreement_gated_residual_reversal_5d'
assert x['validation']['status']=='EFFECTIVE'
assert x['validation']['metrics']['daily_ic']>=.007 and x['validation']['metrics']['daily_icir']>=.084
assert x['validation']['signal_artifact']
print('verified',x['factor_id'],x['validation']['status'],x['validation']['metrics']['daily_ic'],x['validation']['metrics']['daily_icir'],x['last_validated'])
