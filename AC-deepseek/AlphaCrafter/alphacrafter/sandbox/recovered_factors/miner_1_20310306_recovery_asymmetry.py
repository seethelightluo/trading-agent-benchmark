import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms}).sort_index(); r=p.pct_change()
# Recovery asymmetry: assets that recovered strongly from their 60d drawdown, but retain positive 10d trend.
high=p.rolling(60,min_periods=45).max(); dd=p/high-1
rec=(p.pct_change(10)-p.pct_change(30)/3) # recent acceleration, price-only
# Require recovery to be measured relative to each asset's own recent range; cross-sectional demean.
f=(rec + 0.5*dd).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0)
ics={h:[] for h in [1,5,10,20]}; ns={h:[] for h in ics}
for i in range(len(p)-20):
 for h in ics:
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1)],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   ics[h].append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns[h].append(len(q))
for h in ics:
 a=np.array(ics[h]);print('H',h,'dates',len(a),'meanN',round(np.mean(ns[h]),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# regime breakdown at likely selected H10
h=10; out=[]
for i in range(len(p)-h):
 q=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: out.append((p.index[i],spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
v=pd.DataFrame(out,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-08','2031-03')]:
 q=v.loc[a:b,'ic']; print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
rank=f.rank(axis=1,pct=True);print('source_dates',len(p),'instruments',len(syms),'coverage',round(f.notna().mean().mean(),4),'active_dates',f.notna().any(axis=1).sum(),'turnover',round(rank.diff().abs().mean().mean(),4))
print('decay',[(h,round(np.mean(ics[h]),6),round(np.mean(ics[h])/np.std(ics[h],ddof=1),6)) for h in ics])
