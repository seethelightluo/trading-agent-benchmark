import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None and len(d): frames[s]=d.set_index('date')['close'].rename(s)
px=pd.concat(frames.values(),axis=1).sort_index().ffill(); ret=px.pct_change(); r5=px.pct_change(5)
disp=r5.std(axis=1); active=disp>disp.rolling(60,min_periods=40).median()
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20); f=-(r5.sub(r5.median(axis=1),axis=0)).div(vol).where(active,np.nan)
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; a=[]; cov=[]; to=[]
 for i,t in enumerate(f.index):
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
   if i and active.iloc[i] and active.iloc[i-1]:
    x=f.iloc[i].rank(pct=True); p=f.iloc[i-1].rank(pct=True); to.append(np.nanmean(abs(x-p)))
 a=np.array(a); recent=a[-252:]
 print({'horizon':h,'dates':len(a),'mean_assets':round(np.mean(np.array(cov)*15),2),'coverage':round(float(np.mean(cov)),4),'IC':round(float(np.nanmean(a)),5),'ICIR':round(float(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)),5),'hit':round(float(np.mean(a>0)),4),'turnover':round(float(np.nanmean(to)),4),'recent_IC':round(float(np.nanmean(recent)),5),'recent_ICIR':round(float(np.nanmean(recent)/(np.nanstd(recent,ddof=1)+1e-12)),5)})
print('data_dates',px.index.min(),px.index.max(),'assets',len(frames),'active_frac',round(float(active.mean()),4))
