import pandas as pd,numpy as np
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv');q.date=pd.to_datetime(q.date);D[s]=q.sort_values('date').set_index('date').close.rename(s)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
def ix(n):
 q=pd.read_csv('../persistent/index_data/'+n+'.csv');q.date=pd.to_datetime(q.date);return q.sort_values('date').set_index('date').close.reindex(p.index).ffill()
v=ix('VIX'); dx=ix('DXY')
vp=v.pct_change(5); gate=((vp>0.08)&(v.rolling(120,min_periods=60).rank(pct=True)>.60)).astype(float)
# Reversal only after a fresh volatility shock; remove common cross-sectional move.
rr=r.rolling(3,min_periods=3).sum(); cross=rr.sub(rr.mean(axis=1),axis=0); vol=r.rolling(20,min_periods=15).std(); f=(-cross/(vol*np.sqrt(3)+1e-8)*gate.values[:,None]).shift(1)
y=np.log(p).shift(-10)-np.log(p);o=[];ns=[]
for i in range(len(p)):
 a=f.iloc[i].values;b=y.iloc[i].values;ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8:o.append(np.corrcoef(rankdata(a[ok]),rankdata(b[ok]))[0,1]);ns.append(ok.sum())
 else:o.append(np.nan);ns.append(0)
x=pd.DataFrame({'ic':o,'n':ns},index=p.index).loc['2024-01-01':'2033-12-20'].dropna();print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()));print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean());
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=x.loc[a:b];print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
