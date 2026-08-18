import pandas as pd, numpy as np
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv');q.date=pd.to_datetime(q.date);D[s]=q.sort_values('date').set_index('date').close.rename(s)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
def ix(n):
 q=pd.read_csv('../persistent/index_data/'+n+'.csv');q.date=pd.to_datetime(q.date);return q.sort_values('date').set_index('date').close.reindex(p.index).ffill()
v=ix('VIX'); dx=ix('DXY')
# Persistent stress: smoothed VIX percentile, amplified only by sustained positive DXY trend.
vpct=v.rolling(120,min_periods=60).rank(pct=True)
dxy=np.log(dx).diff(20)
stress=(vpct.rolling(5,min_periods=3).mean()*(1+1.5*np.maximum(dxy,0))).clip(0,2)
# Buy recent losers during persistent macro stress, risk adjusted and lagged.
f=(-np.log(p).diff(10)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-8)*stress.values[:,None]).shift(1)
def ev(h):
 y=np.log(p).shift(-h)-np.log(p); out=[]; ns=[]
 for i in range(len(p)):
  a=f.iloc[i].values;b=y.iloc[i].values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append(np.corrcoef(rankdata(a[ok]),rankdata(b[ok]))[0,1]);ns.append(ok.sum())
  else: out.append(np.nan);ns.append(0)
 return pd.DataFrame({'ic':out,'n':ns},index=p.index).loc['2024-01-01':'2033-10-26'].dropna()
x=ev(10);print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=x.loc[a:b];print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
for h in [5,10,20]:print('horizon',h,'IC',ev(h).ic.mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean())
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20331223_stress_persistence_signal.csv',index=False)
