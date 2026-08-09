import pandas as pd
import numpy as np
from pathlib import Path

base=Path('../persistent')
files={
 'continuous':'factor_signals_miner_3_20270225_continuous_regime_reversal.csv',
 'vix_shock':'factor_signals_miner_3_20270225_vix_shock_reversal3.csv',
 'dispersion':'factor_signals_miner_3_20270225_dispersion_reversal.csv',
 'rate_shock':'factor_signals_miner_3_20270225_rate_shock_reversal.csv',
 'breadth':'factor_signals_miner_3_20270225_breadth_intensity_reversal.csv',
}
series={}
for name,fn in files.items():
 p=base/fn
 if not p.exists():
  print('MISSING',name,p); continue
 d=pd.read_csv(p)
 sym='symbol' if 'symbol' in d else 'asset'
 d=d.rename(columns={sym:'symbol'})[['date','symbol','signal']]
 d['date']=pd.to_datetime(d.date)
 d=d.dropna(subset=['signal']).drop_duplicates(['date','symbol'])
 series[name]=d.set_index(['date','symbol']).signal
 print(name,'rows',len(d),'dates',d.date.nunique(),'symbols',d.symbol.nunique())

for a in ['vix_shock','dispersion','rate_shock','breadth']:
 if 'continuous' not in series or a not in series: continue
 x=pd.concat([series['continuous'],series[a]],axis=1,keys=['x','y']).dropna()
 print('CORR',a,'n',len(x),'pearson',x.x.corr(x.y),'spearman',x.x.corr(x.y,method='spearman'))

# Realized cross-sectional signal turnover: rank ordering changes between successive dates.
d=series.get('continuous')
if d is not None:
 wide=d.unstack()
 ranks=wide.rank(axis=1,pct=True)
 common=ranks.dropna(how='all')
 changes=[]
 for i in range(1,len(common)):
  z=pd.concat([common.iloc[i-1],common.iloc[i]],axis=1).dropna()
  if len(z)>=8: changes.append(np.mean(np.abs(z.iloc[:,1]-z.iloc[:,0])>0.2))
 print('TURNOVER_PROXY',np.mean(changes) if changes else None,'observations',len(changes))
# save audit for reproducibility
out=pd.DataFrame({'metric':['note'],'value':['deterministic aligned artifact audit; no new factor admitted']})
out.to_csv(base/'factor_audit_miner_2_20270225_continuous_regime.csv',index=False)
