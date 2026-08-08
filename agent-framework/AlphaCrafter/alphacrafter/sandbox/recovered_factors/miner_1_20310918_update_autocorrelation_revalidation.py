"""Apply 2031-09-18 visibility-safe revalidation to admitted factor record."""
import json
path='factors/miner_1_20310306_residual_return_autocorrelation_expansion_20_60d.json'
x=json.load(open(path,encoding='utf8'))
m={
 'primary_horizon_days':20,'daily_paper_ic':0.047330,'daily_paper_icir':0.139917,
 'ic_std':0.338271,'ic_standard_error':0.009477,'ic_hit_ratio':0.535322,
 'ic_dates':1274,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.113815,
 'signal_cell_coverage':0.317710,'valid_cells':17814,'mean_rank_turnover':0.118216,
 'turnover_dates':1293,'max_abs_library_correlation':0.072747,
 'max_abs_library_correlation_factor':'miner_3_residual_downside_volume_confirmation_deceleration_20_60d',
 'max_abs_library_correlation_common_cells':12591,'library_factors_screened':36,
 'decay':{
  '1d':{'ic':0.011472,'icir':0.034890,'hit_ratio':0.514308,'ic_dates':1293},
  '5d':{'ic':0.030253,'icir':0.093790,'hit_ratio':0.551590,'ic_dates':1289},
  '10d':{'ic':0.039243,'icir':0.120106,'hit_ratio':0.557632,'ic_dates':1284},
  '20d':{'ic':0.047330,'icir':0.139917,'hit_ratio':0.535322,'ic_dates':1274}}}
x['validation']={'period':'2020-01-01 through 2031-09-17; visibility-safe completed-close cutoff','timestamp':'2031-09-18T09:30:00','status':'EFFECTIVE','metrics':m,
 'regime_notes':{'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None},'2025_2026_10d':{'ic_dates':65,'ic':0.116468,'icir':0.465893,'hit_ratio':0.723077},'2027_onward_10d':{'ic_dates':1219,'ic':0.035126,'icir':0.106472,'hit_ratio':0.548811},'interpretation':'The factor clears IC and ICIR gates at 5, 10 and 20 sessions. Its primary 20-session result and post-2027 10-session segment remain positive and above gates; the signal stays highly distinct from screened library signals.'}}
x['last_validated']='2031-09-18T09:30:00';x['validation_timestamp']='2031-09-18T09:30:00'
x.setdefault('benchmark_admission',{}).setdefault('selected_metrics',{}).update({'ic':0.047330,'icir':0.139917,'metric_path':'validation.metrics (20d)','max_abs_library_correlation':0.072747,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.00662277361})
open(path,'w',encoding='utf8').write(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print('updated',path,'status',x['validation']['status'],'quality',x['benchmark_admission']['selected_metrics']['quality'])
