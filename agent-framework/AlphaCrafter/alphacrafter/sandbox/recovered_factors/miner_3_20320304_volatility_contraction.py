import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; q={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); q[a]=d.set_index('date').close
p=pd.DataFrame(q).sort_index(); r=p.pct_change()
# Volatility contraction followed by continuation: inverse 20d/60d realized-vol ratio.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); fac=-(v20/(v60+1e-12))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; zics=[]; ns=[]
 for dt in p.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(zics);print(f'H{h} dates={len(x)} meanN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/(x.std(ddof=1)+1e-12):.6f} hit={np.mean(x>0):.3f}')
print('coverage=%.4f assets=%d dates=%d'%(fac.notna().sum().sum()/fac.size,len(A),len(p)))
for lab,sub in [('2020-23',p.index[p.index.year<=2023]),('2024-27',p.index[(p.index.year>=2024)&(p.index.year<=2027)]),('2028-30',p.index[(p.index.year>=2028)&(p.index.year<=2030)]),('2031+',p.index[p.index.year>=2031]),('recent120',p.index[-120:])]:
 x=[]
 for dt in sub:
  z=pd.concat([fac.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print(lab,'n=',len(x),'IC=%.6f ICIR=%.6f'%(x.mean(),x.mean()/(x.std(ddof=1)+1e-12)))
# 10-day rank turnover
rk=fac.rank(axis=1,pct=True); t=[]
for i in range(10,len(rk),10):
 x=rk.iloc[i-10];y=rk.iloc[i];c=x.dropna().index.intersection(y.dropna().index)
 if len(c)>=8:t.append(np.mean(abs(x[c]-y[c])))
print('turnover10=%.4f'%np.mean(t))
