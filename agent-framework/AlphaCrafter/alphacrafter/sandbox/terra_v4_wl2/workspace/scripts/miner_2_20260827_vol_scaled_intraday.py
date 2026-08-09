import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:'2026-07-15']
  D[s]=x
 except Exception as e: print('missing',s)
# one-day intraday reversal normalized by trailing 20d close-return volatility; all values date aligned
for variant in ['scaled','winsor_scaled']:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); intr=x.open/x.close-1; vol=r.rolling(20,min_periods=15).std()
  f=intr/(vol+1e-8)
  if variant=='winsor_scaled': f=f.clip(f.rolling(120,min_periods=30).quantile(.05),f.rolling(120,min_periods=30).quantile(.95))
  for i,dt in enumerate(x.index):
   if pd.notna(f.iloc[i]) and i+1<len(x): rows.append((dt,s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); ic=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ic.append(spearmanr(g.f,g.y).statistic)
 z=np.array(ic);
 if variant=='winsor_scaled': a.to_csv('../persistent/factor_signals_miner_2_20260827_vol_scaled_intraday.csv',index=False)
 print(variant,'dates',len(z),'avg_names',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',a.groupby('date').apply(lambda g:g.s.iloc[np.argsort(g.f.values).tolist()].count()).mean() if False else 'na')
 for yr in range(2020,2027):
  q=[]
  for dt,g in a.groupby('date'):
   if dt.year==yr and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  if q: print(yr,round(float(np.mean(q)),5),round(float(np.mean(q)/np.std(q,ddof=1)),4),len(q))
