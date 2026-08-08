"""Apply 2031-07-10 visibility-safe revalidation evidence to the admitted factor."""
import json
path='factors/miner_1_20310417_residual_positive_jump_concentration_expansion_20_60d.json'
d=json.load(open(path,encoding='utf8'))
m=d['validation']['metrics']
m.update({
 'primary_horizon_days':20,'daily_paper_ic':0.030727,'daily_paper_icir':0.093205,
 'ic_std':0.329666,'ic_standard_error':0.009489,'ic_hit_ratio':0.518641,
 'ic_dates':1207,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.013256,
 'signal_cell_coverage':0.307574,'mean_rank_turnover':0.123137,'turnover_dates':1226,
 'max_abs_library_correlation':0.427951,
 'max_abs_library_correlation_factor':'miner_3_risk_adjusted_trend_20d',
 'max_abs_library_correlation_common_cells':17009,'library_factors_screened':35,
 'decay':{
   '1d':{'ic':0.012073,'icir':0.036711,'hit_ratio':0.518760,'ic_dates':1226,'mean_valid_instruments':10.013051},
   '5d':{'ic':0.000783,'icir':0.002369,'hit_ratio':0.496727,'ic_dates':1222,'mean_valid_instruments':10.013093},
   '10d':{'ic':-0.001934,'icir':-0.005891,'hit_ratio':0.493016,'ic_dates':1217,'mean_valid_instruments':10.013147},
   '20d':{'ic':0.030727,'icir':0.093205,'hit_ratio':0.518641,'ic_dates':1207,'mean_valid_instruments':10.013256}
 }
})
d['validation'].update({
 'period':'2020-01-01 through 2031-07-09; visibility-safe completed-close cutoff',
 'timestamp':'2031-07-10T09:30:00','status':'EFFECTIVE',
 'regime_notes':{
  '2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},
  '2025_2026_10d':{'ic_dates':48,'ic':0.083421,'icir':0.267722,'hit_ratio':0.583333},
  '2027_onward_10d':{'ic_dates':1169,'ic':-0.005438,'icir':-0.016552,'hit_ratio':0.489307},
  'interpretation':'The pre-specified 20-session primary horizon remains above both shared gates (absolute IC 0.0070; absolute ICIR 0.0840), but has weakened from the prior validation. The post-2027 10-session diagnostic is mildly negative. Retained as a low-weight conditional medium-horizon diversifier, with expedited next revalidation; not suitable as a short-horizon standalone signal.'
 }
})
d['last_validated']=d['validation_timestamp']='2031-07-10T09:30:00'
# Preserve original admission provenance, while reflecting latest valid selected evidence.
d['benchmark_admission']['selected_metrics'].update({'ic':0.030727,'icir':0.093205,'metric_path':'validation.metrics (20d primary)','max_abs_library_correlation':0.427951,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.0028639})
open(path,'w',encoding='utf8').write(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
print('updated',path,'status',d['validation']['status'],'primary',m['daily_paper_ic'],m['daily_paper_icir'],'corr',m['max_abs_library_correlation'])
