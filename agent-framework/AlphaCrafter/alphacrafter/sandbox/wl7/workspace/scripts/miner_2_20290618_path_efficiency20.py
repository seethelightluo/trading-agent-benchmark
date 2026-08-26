import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

files=glob.glob('../persistent/stock_data/*.csv')
prices={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); prices[s]=d
close=pd.DataFrame({s:d.close for s,d in prices.items()}).sort_index()
ret=close.pct_change()
# Path-efficiency trend: signed 20d return divided by total absolute daily movement.
# lagged by one day at each observation; forward 10d return.
raw=close/close.shift(20)-1
path=raw/(ret.abs().rolling(20).sum()+1e-12)
factor=path.shift(1)
fwd=close.shift(-10)/close-1
rows=[]; sig=[]
for dt in close.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
  for s in z.index: sig.append((dt,s,float(x[s])))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=r.ic.mean(); sd=r.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('factor=path_efficiency20 dates=%d avg_n=%.2f coverage=%.2f IC=%.6f daily_ICIR=%.6f hit=%.4f'%(len(r),r.n.mean(),r.n.mean()/15,mean,icir,(r.ic>0).mean()))
for name,mask in [('2020-2024',(r.index<'2025-01-01')),('2025-2026',((r.index>='2025-01-01')&(r.index<'2027-01-01'))),('2027-2028',((r.index>='2027-01-01')&(r.index<'2029-01-01'))),('since2028-09',(r.index>='2028-09-01'))]:
 q=r.loc[mask]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
# signal artifact for provenance
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290618_path_efficiency20_signal.csv',index=False)
