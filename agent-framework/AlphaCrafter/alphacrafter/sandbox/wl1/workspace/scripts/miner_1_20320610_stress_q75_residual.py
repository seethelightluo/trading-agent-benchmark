import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().astype(float)
r=np.log(p).diff(); b=r.mean(axis=1); res=r.sub(b,axis=0); disp=res.std(axis=1)
active=(b.rolling(20).sum()<0)|(disp>disp.rolling(120,min_periods=80).quantile(.75))
f=(-res.rolling(10).sum().div(res.rolling(60).std())).where(active).shift(1)
ics=[];ds=[];ns=[]
for d in f.index:
 y=np.log(p.shift(-10)/p).loc[d];x=f.loc[d];ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(q):ics.append(q);ds.append(d);ns.append(ok.sum())
a=np.array(ics);print('factor=stress_or_q75_residual_pullback_10d');print('dates',len(a),'calendar',len(p),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for aa,bb in [('2024','2026'),('2027','2029'),('2030','2032')]:
 z=a[(np.array(ds)>=pd.Timestamp(aa+'-01-01'))&(np.array(ds)<=pd.Timestamp(bb+'-12-31'))];print(aa+'-'+bb,len(z),z.mean(),z.mean()/z.std(ddof=1))
for h in [1,5,20]:
 y=np.log(p.shift(-h)/p);z=[]
 for d in f.index:
  x=f.loc[d];yy=y.loc[d];ok=x.notna()&yy.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],yy[ok]).statistic)
 print('horizon',h,'IC',np.mean(z),'n',len(z))
pd.DataFrame(f).to_csv('scripts/miner_1_20320610_stress_q75_residual_signal.csv')
