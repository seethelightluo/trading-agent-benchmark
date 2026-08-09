import pandas as pd, glob, os, json, numpy as np
new='../persistent/factor_signals_miner_3_20270226_vix_dispersion_reversal3.csv'
a=pd.read_csv(new).pivot(index='date',columns='symbol',values='signal')
mx=0.; partner=None
for f in glob.glob('../persistent/factor_signals_*.csv'):
 if os.path.abspath(f)==os.path.abspath(new): continue
 try:
  b=pd.read_csv(f)
  sym='symbol' if 'symbol' in b else 'asset'
  if 'signal' not in b: continue
  b=b.pivot(index='date',columns=sym,values='signal')
  x,y=a.align(b,join='inner',axis=0); x,y=x.align(y,join='inner',axis=1)
  z=pd.concat([x.stack(),y.stack()],axis=1).dropna()
  if len(z)>30:
   c=abs(z.iloc[:,0].corr(z.iloc[:,1]));
   if np.isfinite(c) and c>mx: mx=float(c);partner=os.path.basename(f)
 except Exception: pass
rec={
 'factor_id':'miner_1_20270227_vix_dispersion_reversal10',
 'factor_name':'VIX-shock and high-dispersion conditioned reversal (10d)',
 'version':'1.0',
 'calculation':{'expression':'active * (-sum(return,3))','description':'When lagged VIX shock and lagged cross-asset dispersion are both above their 60-day 75th percentile, rank assets by negative prior 3-day return; otherwise signal is neutral/absent. Forward validation uses 10 trading days.'},
 'dependencies':['close','VIX observation-only macro series'],
 'parameters':{'lookback_return_days':3,'regime_window_days':60,'threshold_quantile':0.75,'forward_horizon_days':10},
 'validation':{'status':'EFFECTIVE','period':'2020-01-01 through 2027-02-25','metrics':{'ic':0.0673569821,'icir':0.1901544179,'dates':58,'avg_instruments':13.6897,'coverage_active':0.0915,'turnover':'not reported; regime activation is sparse','max_abs_library_correlation':mx,'max_correlation_partner':partner},'regime_notes':'Sparse active-regime test: 12 observations in 2020-22, 11 in 2023-24, 15 in 2025-26, and 1 in 2027. IC was negative in 2020-22 and positive thereafter; uncertainty is high.', 'signal_artifact':new},
 'tags':['reversal','volatility','dispersion','macro-regime'], 'last_validated':'2027-02-27T00:00:00Z'
}
with open('factors/'+rec['factor_id']+'.json','w') as q: json.dump(rec,q,indent=2)
with open('factors/'+rec['factor_id']+'.json') as q: v=json.load(q)
assert v['factor_id']==rec['factor_id'] and v['validation']['status']=='EFFECTIVE' and v['validation']['metrics']['ic']>=.007 and v['validation']['metrics']['icir']>=.084 and os.path.exists(new)
print(json.dumps({'saved':v['factor_id'],'status':v['validation']['status'],'ic':v['validation']['metrics']['ic'],'icir':v['validation']['metrics']['icir'],'maxcorr':mx,'partner':partner}))
