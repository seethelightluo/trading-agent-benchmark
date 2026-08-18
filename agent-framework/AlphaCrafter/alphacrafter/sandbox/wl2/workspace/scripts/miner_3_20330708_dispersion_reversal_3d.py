import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index()
d={s:ld(s) for s in U};c=pd.concat({s:x.close for s,x in d.items()},axis=1);r=np.log(c).diff();v20=r.rolling(20,min_periods=12).std()
base=-r.rolling(3).sum()/(v20*np.sqrt(3)); cross_disp=r.std(axis=1).rolling(20,min_periods=12).rank(pct=True)
f=(base.mul(cross_disp,axis=0)).shift(1); rows=[]
for dt in r.index:
 x=f.loc[dt];y=r.shift(-1).loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(z),'assets',15,'coverage',z.n.mean()/15,'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for a,b in [('2026','2029-12-31'),('2030','2033-07-08')]:
 q=z.loc[a:b];print('regime',a,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
for h in [3,5,10]:
 y=r.rolling(h).sum().shift(-h);q=[]
 for dt in r.index:
  x=f.loc[dt];yy=y.loc[dt];ok=x.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],yy[ok]).statistic)
 print('decay',h,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
print('turnover',np.mean(np.sign(f).diff().abs().stack()>0));f.to_csv('scripts/miner_3_20330708_dispersion_reversal_signal.csv')
