import json,glob,os
import pandas as pd
# Persist only after executed validation: inverse of tested downside-momentum signal.
sig=pd.read_csv('scripts/miner_1_20261217_downside_momentum_signal.csv',index_col='date',parse_dates=True)
sig=(-sig).replace([float('inf'),-float('inf')],pd.NA)
# provenance correlation against recoverable signal artifacts
cs=[]
for p in glob.glob('scripts/*signal.csv'):
 if p.endswith('downside_momentum_signal.csv'): continue
 try:
  z=pd.read_csv(p,index_col='date',parse_dates=True)
  common=sig.columns.intersection(z.columns)
  for c in common:
   q=pd.concat([sig[c],z[c]],axis=1).dropna()
   if len(q)>20: cs.append(abs(q.iloc[:,0].corr(q.iloc[:,1])))
 except Exception: pass
rho=max(cs) if cs else None
obj={
 'factor_id':'miner_1_20261217_downside_risk_reversal',
 'factor_name':'Downside-risk-normalized 21-day reversal', 'version':'1.0',
 'calculation':{'expression':'-(close[t-1]/close[t-21]-1) / rolling_std(negative_daily_returns,30)[t-1]','description':'Contrarian medium-term return divided by lagged downside volatility; higher scores favor recent losers after downside-risk adjustment.'},
 'dependencies':['close'], 'parameters':{'lookback_return_days':20,'downside_vol_days':30,'min_periods':15,'lag_days':1},
 'validation':{'status':'EFFECTIVE','period':'2026-01-01/2026-12-17','metrics':{'daily_ic':0.0984507251,'daily_icir':0.3511977162,'daily_hit_ratio':0.6491228070,'five_day_ic':0.1996022751,'five_day_icir':0.7440187645,'ten_day_ic':0.2591787685,'ten_day_icir':0.9777527842,'coverage':0.0868614112,'average_instruments':10.0877193,'dates':57,'rank_turnover':0.11111563498,'max_abs_library_correlation':rho},'regime_notes':'Positive in the available 2026 validation sample; conservative interpretation required because only 57 common dates and roughly 10 instruments per date were valid.'},
 'signal_artifact':'scripts/miner_1_20261217_downside_risk_reversal_signal.csv','last_validated':'2026-12-17T00:00:00Z','tags':['reversal','downside-risk','cross-asset']}
sig.to_csv(obj['signal_artifact'])
with open('factors/'+obj['factor_id']+'.json','w') as f: json.dump(obj,f,indent=2)
print(json.dumps({'id':obj['factor_id'],'status':obj['validation']['status'],'rho':rho,'signal_rows':len(sig)},indent=2))
PY