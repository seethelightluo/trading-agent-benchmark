import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-18')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); m=r.median(axis=1)
# idiosyncratic reversal: reverse recent return after removing contemporaneous cross-asset median move
for L in [2,3,5,10]:
 f=-(r.sub(m,axis=0)).rolling(L,min_periods=L).sum(); rows={h:[] for h in [1,5,10]}
 for i in range(15,len(p)-10):
  for h in rows:
   z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('L',L,'coverage',round(f.notna().sum().sum()/f.size,4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
 for h,q in rows.items():
  q=np.array(q); print(' h',h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('range',p.index.min(),p.index.max(),'n',len(p))
