import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None: fs[s]=d.set_index('date')['close'].rename(s)
p=pd.concat(fs.values(),axis=1).sort_index().ffill(); r=p.pct_change(); r5=p.pct_change(5); disp=r5.std(1); q=disp.rolling(120,min_periods=80).quantile(.75); on=disp>q
v=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=-(r5.sub(r5.median(1),axis=0)).div(v).where(on,np.nan)
for h in [5,10,20]:
 y=p.shift(-h)/p-1; ic=[]; cv=[]; tr=[]
 for i,t in enumerate(f.index):
  z=pd.concat([f.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));cv.append(len(z)/15)
   if i and on.iloc[i] and on.iloc[i-1]: tr.append(np.nanmean(abs(f.loc[t].rank(pct=True)-f.iloc[i-1].rank(pct=True))))
 a=np.array(ic); rr=a[-252:]
 print({'h':h,'dates':len(a),'assets':round(np.mean(np.array(cv)*15),2),'coverage':round(np.mean(cv),4),'IC':round(np.mean(a),5),'ICIR':round(np.mean(a)/np.std(a,ddof=1),5),'hit':round(np.mean(a>0),4),'turnover':round(np.mean(tr),4),'recent_IC':round(np.mean(rr),5),'recent_ICIR':round(np.mean(rr)/np.std(rr,ddof=1),5)})
print('period',p.index.min(),p.index.max(),'assets',len(fs),'active',round(on.mean(),4))
