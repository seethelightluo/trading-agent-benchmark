import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-09-06')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cutoff].sort_values('date').set_index('date'); P[s]=d.close.astype(float)
P=pd.DataFrame(P).sort_index(); R=P.pct_change()
factor=-(P.pct_change(3).shift(1)).div(R.rolling(20).std().shift(1))
def calc(h):
 out=[]; ns=[]
 for i in range(len(P)-h):
  f=factor.iloc[i]; fr=P.iloc[i+h]/P.iloc[i]-1; ok=f.notna()&fr.notna()
  if ok.sum()>=8: out.append(spearmanr(f[ok],fr[ok]).statistic); ns.append(ok.sum())
 return np.array(out),np.array(ns)
for h in [1,5,10]:
 a,n=calc(h); print('H',h,'dates',len(a),'avg_n',n.mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean())
a,n=calc(1); print('coverage',factor.loc[:cutoff].notna().sum(axis=1).mean()/15,'turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-09-06')]:
 dates=P.index[(P.index>=lo)&(P.index<=hi)]; vals=[]
 for dt in dates:
  i=P.index.get_loc(dt)
  if i>=len(P)-1: continue
  f=factor.loc[dt]; fr=R.iloc[i+1]; ok=f.notna()&fr.notna()
  if ok.sum()>=8: vals.append(spearmanr(f[ok],fr[ok]).statistic)
 print(name,len(vals),np.mean(vals) if vals else np.nan)
print('period',P.index.min(),P.index.max(),'assets',P.shape[1])
