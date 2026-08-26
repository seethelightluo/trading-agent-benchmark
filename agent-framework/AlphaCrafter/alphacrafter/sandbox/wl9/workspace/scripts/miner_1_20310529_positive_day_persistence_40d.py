import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-05-29')
d={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 d[s]=x.loc[:END,'close']
p=pd.DataFrame(d).sort_index()
r=p.pct_change()
# persistence: excess fraction of positive daily returns, conditioned by medium trend magnitude
f=(r.gt(0).rolling(40,min_periods=30).mean()-0.5) * (p.pct_change(40))
# signal known at t, forward return begins t+1
out=[]
for h in [5,10,20,40]:
  vals=[]
  for i in range(len(p)-h):
   a=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1
   z=pd.concat([a,y.rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.y).statistic)
  vals=np.asarray(vals); ic=np.nanmean(vals); sd=np.nanstd(vals,ddof=1)
  print(h,'dates',len(vals),'avgN',round(np.nanmean([len(pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()) for i in range(len(p)-h)]),2),'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252),6),'hit',round(np.mean(vals>0),4))
# turnover proxy and regime 40
vals=[]
for i in range(len(p)-40):
 z=pd.concat([f.iloc[i],(p.iloc[i+40]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8: vals.append((p.index[i],spearmanr(z.iloc[:,0],z.y).statistic))
v=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-05-29')]:
 q=v.loc[a:b].ic; print('regime',a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),6) if len(q)>1 else np.nan)
# artifact
sig=f.loc[v.index].copy(); sig.index.name='date'; sig.to_csv('scripts/miner_1_20310529_positive_day_persistence_signal.csv')
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
