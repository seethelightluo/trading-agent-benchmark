import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,5000)
   if d is not None and len(d): return d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
  except Exception: pass
P={s:load(s) for s in U}; px=pd.DataFrame({s:x for s,x in P.items() if x is not None}).sort_index().ffill(limit=3); r=np.log(px).diff()
# Candidate: normalized drawdown rebound. Buy assets with deep, recent drawdowns,
# but only during broad stress, when snapback odds may be higher than ordinary trend.
vol=r.rolling(40).std().clip(lower=1e-5)
dd=px/px.rolling(60).max()-1
stress=(r<0).rolling(10).mean().mean(axis=1)
gate=((stress-.5).clip(lower=0)/.5).clip(0,1)
f=(-dd.div(vol)).mul(gate,axis=0).shift(1)
fr=px.shift(-10)/px-1
rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna();m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('factor=stress_drawdown_rebound20');print('dates',len(x),'avg_n',x.n.mean(),'coverage',f.notna().sum().sum()/f.size,'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(x.ic>0).mean());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 y=x.loc[a:b].ic;print(a,len(y),y.mean(),y.mean()/y.std(ddof=1)*np.sqrt(252) if len(y)>2 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20351224_stress_drawdown_rebound_signal.csv',index=False)
