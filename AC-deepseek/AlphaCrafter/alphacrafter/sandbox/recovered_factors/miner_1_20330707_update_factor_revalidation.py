"""Update validated Miner-1 factor record after 2033-07-07 scheduled revalidation."""
import json
p='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
with open(p,encoding='utf8') as fh: d=json.load(fh)
d['version']='revalidated_2033-07-07'
d['last_validated']='2033-07-07T09:30:00'
v=d['validation']
v['period']='2020-01-01 through 2033-07-06; visibility-safe completed-close cutoff'
v['timestamp']='2033-07-07T09:30:00'
v['status']='EFFECTIVE'
v['metrics']={
 'primary_horizon_days':20,'daily_paper_ic':0.055755,'daily_paper_icir':0.162154,
 'ic_std':0.343841,'ic_standard_error':0.008274,'ic_hit_ratio':0.564563,
 'ic_dates':1727,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.011581,
 'signal_cell_coverage':0.352012,'valid_cells':22219,'mean_rank_turnover':0.211190,
 'turnover_dates':1746,'library_factors_screened':34,
 'max_abs_library_correlation':0.401416,
 'max_abs_library_correlation_factor':'miner_3_residual_upside_volume_confirmation_60d',
 'max_abs_library_correlation_common_cells':15218,
 'library_correlation_evidence_complete':True,
 'decay':{
  '1d':{'ic':0.014704,'icir':0.044599,'hit_ratio':0.508018,'ic_dates':1746},
  '5d':{'ic':0.033029,'icir':0.102880,'hit_ratio':0.540184,'ic_dates':1742},
  '10d':{'ic':0.041387,'icir':0.126126,'hit_ratio':0.544617,'ic_dates':1737},
  '20d':{'ic':0.055755,'icir':0.162154,'hit_ratio':0.564563,'ic_dates':1727}
 }}
v['regime_notes']={
 '2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},
 '2025_2026_10d':{'ic_dates':48,'ic':0.009275,'icir':0.029474,'hit_ratio':0.541667},
 '2027_onward_10d':{'ic_dates':1689,'ic':0.042300,'icir':0.128743,'hit_ratio':0.544701},
 'interpretation':'Revalidation passes binding gates at 5, 10, and 20 days, strongest at 20 days. Full-library Spearman evidence is present and remains below the 0.5000 ceiling. The 2027-onward segment remains positive; the short 2025-26 segment is weak. Mean IC cross-section is 10.01 instruments, exceeding the eight-instrument minimum.'}
with open(p,'w',encoding='utf8') as fh: json.dump(d,fh,indent=2,ensure_ascii=False);fh.write('\n')
print('updated',p,'status',v['status'],'primary',v['metrics']['daily_paper_ic'],v['metrics']['daily_paper_icir'])
