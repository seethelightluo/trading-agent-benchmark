import pandas as pd,numpy as np
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U: xs.append(pd.read_csv('../persistent/stock_data/'+s+'.csv',usecols=['date','close'],parse_dates=['date']).set_index('date')['close'])
pd0=pd.concat(xs,axis=1,keys=U).sort_index(); r=np.log(pd0).diff().to_numpy(float); dates=pd0.index
# medium-term continuation excluding recent short-term: 60d return - 10d return; and stress-sensitive low vol
cands={'medium_minus_short':lambda i,j: np.nansum(r[i-59:i+1,j])-np.nansum(r[i-9:i+1,j]),'lowvol':lambda i,j:-np.nanstd(r[i-19:i+1,j])}
for name,fn in cands.items():
 for H in [10]:
  ic=[]; ns=[]
  for i in range(100,len(r)-H):
   f=[fn(i,j) for j in range(15)]; y=[np.nansum(r[i+1:i+1+H,j]) for j in range(15)]; ok=np.isfinite(f)&np.isfinite(y)
   if ok.sum()>=8: ic.append(np.corrcoef(rankdata(np.array(f)[ok]),rankdata(np.array(y)[ok]))[0,1]);ns.append(ok.sum())
  a=np.array(ic); print(name,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),5),'hit',round(np.mean(a>0),3))
  for n in [250,500]:
   q=a[-n:];print(' recent',n,round(q.mean(),5),round(q.mean()/(q.std(ddof=1)+1e-12),5),round(np.mean(q>0),3))
