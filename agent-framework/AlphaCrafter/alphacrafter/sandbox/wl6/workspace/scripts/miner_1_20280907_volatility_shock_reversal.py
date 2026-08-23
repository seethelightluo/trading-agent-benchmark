import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2028-09-06')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date');P[s]=d.close.astype(float)
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); v5=r.rolling(5).std();v60=r.rolling(60).std(); f=-(v5/v60-1).shift(1)
def run(h):
 z=[];n=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic);n.append(ok.sum())
 a=np.array(z);return a,np.array(n)
for h in [1,5,10]:
 a,n=run(h);print('H',h,'dates',len(a),'avg_n',n.mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean())
a,n=run(1);print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,lo,hi in [('2025-26','2025','2026-12-31'),('2027-28','2027','2028-09-06')]:
 vals=[]
 for dt in P.index[(P.index>=lo)&(P.index<=hi)]:
  i=P.index.get_loc(dt);
  if i>=len(P)-1: continue
  x=f.loc[dt];y=r.iloc[i+1];ok=x.notna()&y.notna()
  if ok.sum()>=8:vals.append(spearmanr(x[ok],y[ok]).statistic)
 print(name,len(vals),np.mean(vals))
print('period',P.index.min(),P.index.max(),'assets',P.shape[1])
