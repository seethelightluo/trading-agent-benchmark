import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-23'); ps={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);d=d[d.date<=end].sort_values('date').set_index('date');ps[s]=d.close.astype(float)
p=pd.DataFrame(ps).sort_index(); rr=p.pct_change(); v20=rr.rolling(20,min_periods=15).std(); v60=rr.rolling(60,min_periods=40).std(); f=-(v20/v60); y=p.shift(-10)/p-1
for name,fac in [('low_vol',f),('vol_change',-(v20-v60)/v60)]:
 a=[]; ns=[]; ts=[]; last=None
 for dt in fac.index:
  x=fac.loc[dt];z=y.loc[dt];ok=x.notna()&z.notna()
  if ok.sum()>=8:
   a.append(spearmanr(x[ok],z[ok]).statistic);ns.append(ok.sum());rk=x.rank(pct=True);ts.append(np.nan if last is None else np.mean(abs(rk-last)));last=rk
 a=np.array(a);print(name,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(ts),4))
 for h in [1,5,10,20]:
  yy=p.shift(-h)/p-1;q=[]
  for dt in fac.index:
   ok=fac.loc[dt].notna()&yy.loc[dt].notna()
   if ok.sum()>=8:q.append(spearmanr(fac.loc[dt][ok],yy.loc[dt][ok]).statistic)
  q=np.array(q);print(' decay',h,round(np.mean(q),6),round(np.mean(q)/np.std(q,ddof=1),6))
