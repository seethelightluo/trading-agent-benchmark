import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x['date']); D[s]=x.sort_values('date').set_index('date')
def calc(h):
 rows=[]
 for s,x in D.items():
  prev=x.close.shift(1); gap=np.log(x.open/prev); intr=np.log(x.close/x.open); vol=intr.rolling(20).std().shift(1)
  f=-gap/(vol+1e-8); y=np.log(x.close.shift(-h)/x.close)
  rows.append(pd.DataFrame({'date':x.index,'s':s,'f':f.values,'y':y.values}).dropna())
 z=pd.concat(rows,ignore_index=True); out=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8:
   q=spearmanr(g.f,g.y).statistic
   if np.isfinite(q): out.append((dt,q,len(g)))
 a=pd.DataFrame(out,columns=['date','ic','n']); return z,a
z,a=calc(1); print('dates',len(a),'avgN',a.n.mean(),'coverage',len(z)/sum(len(x) for x in D.values())); print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for name,m in [('2020-22',a.date<'2023-01-01'),('2023-24',(a.date>='2023-01-01')&(a.date<'2025-01-01')),('2025-27',a.date>='2025-01-01')]:
 b=a[m]; print(name,len(b),b.ic.mean(),b.ic.mean()/b.ic.std(ddof=1))
for h in [5,10]:
 _,b=calc(h); print('h',h,'IC',b.ic.mean(),'ICIR',b.ic.mean()/b.ic.std(ddof=1),'dates',len(b))
z.to_csv('scripts/miner_3_20270325_gap_reversal_signal.csv',index=False)
