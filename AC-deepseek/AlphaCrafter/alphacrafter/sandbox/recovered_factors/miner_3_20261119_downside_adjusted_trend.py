import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close
p=pd.concat([L(a).rename(a) for a in A],axis=1).sort_index().ffill(); r=p.pct_change()
# downside-risk-adjusted trend: 20d compounded/simple return divided by downside deviation of daily returns
mom=r.rolling(20,min_periods=15).sum(); dn=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
sig=mom/dn.replace(0,np.nan)
print('period',p.index.min().date(),p.index.max().date(),'dates',len(p),'assets',len(A))
for h in [1,5,10,20]:
 f=p.pct_change(h).shift(-h);z=[];ds=[];ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic
   if np.isfinite(q):z.append(q);ds.append(d);ns.append(ok.sum())
 z=pd.Series(z,index=ds);print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==10:
  for y,g in z.groupby(z.index.year):print(' year',y,'IC %.6f n %d'%(g.mean(),len(g)))
print('coverage %.4f turnover %.4f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
# compare library factor proxies
for name,x in [('trend',mom/r.rolling(20,min_periods=15).std()),('reversal',-r.rolling(5,min_periods=4).sum()),('invvol',-r.rolling(20,min_periods=15).std())]:
 xx=[];yy=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&x.loc[d].notna();xx+=sig.loc[d,ok].tolist();yy+=x.loc[d,ok].tolist()
 print('corr',name,'rho %.6f cells %d'%(spearmanr(xx,yy).statistic,len(xx)))
