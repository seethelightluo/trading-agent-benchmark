import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={};
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);D[s]=d.set_index('date')
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); V=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).reindex(P.index)
r=np.log(P).diff(); rv=r.rolling(5).sum(); volshock=(V/(V.rolling(20).median()+1e-12)).clip(0,10)
# Reversal is strengthened by abnormal volume, lagged one day.
f=(-rv*volshock).shift(1)
def run(h):
 a=[]
 for i in range(len(P)-h):
  x=f.iloc[i];y=P.iloc[i+h]/P.iloc[i]-1;ok=x.notna()&y.notna()
  if ok.sum()>=8:a.append((P.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
 return pd.DataFrame(a,columns=['date','ic','n']).set_index('date')
z=run(10);print('universe',15,'dates',len(z),'avg_names',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q=run(h);print('decay',h,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'dates',len(q))
for a,b in [('2025-01-01','2027-12-31'),('2028-01-01','2030-05-01')]:
 q=z.loc[a:b];print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
z.to_csv('scripts/miner_3_20300520_volume_reversal_ic.csv');f.to_csv('scripts/miner_3_20300520_volume_reversal_signal.csv')
