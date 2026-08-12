import json
fid='miner_2_20310904_vix_dispersion_p60_reversal_5d'
obj={
 'factor_id':fid,
 'factor_name':'VIX-confirmed elevated-dispersion 5-day reversal',
 'version':'2031-09-04',
 'calculation':{
  'expression':'-ret_5d(asset) when cross_sectional_std(ret_5d) > rolling_median_252(cross_sectional_std(ret_5d)) and pct_change(VIX,5) > 0; else 0',
  'description':'Contrarian five-day return signal activated when cross-asset dispersion is above its trailing median and observation-only VIX is rising over five sessions.'
 },
 'dependencies':['close','VIX close'],
 'parameters':{'return_window':5,'dispersion_window':252,'dispersion_threshold':'rolling median','vix_change_window':5,'min_cross_section':8},
 'validation':{
  'status':'EFFECTIVE',
  'period':'2020-01-01 through 2031-09-03; forward 1-day returns',
  'last_validated':'2031-09-04T00:00:00Z',
  'metrics':{
   'daily_paper_ic':0.044332764,
   'daily_paper_icir':0.129066059,
   'ic_dates':3738,
   'average_instruments':14.76297,
   'coverage':0.984198,
   'active_rate':0.216159,
   'turnover':0.0900,
   'hit_ratio':'see artifact; constant-signal dates excluded',
   'max_abs_library_correlation':None
  },
  'regime_notes':'IC/ICIR: 2020-22 +0.10159/+0.27622 (592 dates); 2023-25 +0.06016/+0.16111 (580); 2026-31 +0.02083/+0.06362 (1440). Full-history gate passes, but recent regime drift warning.'
 },
 'signal_artifact':'scripts/miner_2_20310904_vix_dispersion_reversal_signal.csv',
 'tags':['reversal','dispersion','volatility-regime','macro-conditioned','cross-asset']
}
with open('factors/'+fid+'.json','w') as f: json.dump(obj,f,indent=2)
print(json.load(open('factors/'+fid+'.json'))['factor_id'],json.load(open('factors/'+fid+'.json'))['validation']['status'],json.load(open('factors/'+fid+'.json'))['validation']['metrics']['daily_paper_ic'],json.load(open('factors/'+fid+'.json'))['signal_artifact'])
