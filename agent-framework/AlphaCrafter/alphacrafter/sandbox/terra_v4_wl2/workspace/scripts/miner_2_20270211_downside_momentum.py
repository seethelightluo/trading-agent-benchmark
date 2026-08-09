import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2027-02-10'
frames={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date); d=d[d.date<=end].set_index('date'); frames[s]=d
px=pd.concat({s:f.close for s,f in frames.items()},axis=1).sort_index()
rets=px.pct_change()
# downside-risk-adjusted medium momentum: trailing 20d return divided by downside deviation, lagged one day
variants={}
for w in [20,40,60]:
 down=rets.where(rets<0).rolling(w,min_periods=max(10,w//2)).std()
 variants[f'damom_{w}']=px.pct_change(w)/down
# evaluate against forward close return, only dates with >=8 assets
for name,x in variants.items():
 x=x.shift(1); fwd=rets.shift(-1)
 vals=[]; dates=[]; cov=[]; turns=[]; prev=None
 for dt in x.index:
  z=pd.concat([x.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); cov.append(len(z)/len(U))
   r=x.loc[dt].rank(); turns.append(np.nan if prev is None else (r-prev).abs().sum()/(len(U)**2)); prev=r
 ic=pd.Series(vals,index=dates).dropna(); mean=ic.mean(); sd=ic.std(ddof=1); ir=mean/sd*np.sqrt(252) if sd else np.nan
 print(name,'dates',len(ic),'avgN',np.mean(np.array(cov)*len(U)),'IC',round(mean,8),'ICIR',round(ir,8),'hit',round((ic>0).mean(),4),'turn',round(np.nanmean(turns),4))
 for label,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('recent','2026-07-16',end)]:
  q=ic[(ic.index>=a)&(ic.index<=b)]; print(' ',label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),4) if len(q)>2 else np.nan)
 for h in [5,10]:
  ff=px.pct_change(h).shift(-h)/px.shift(-h)*0+px.pct_change(h).shift(-h) # forward h return
  vv=[]
  for dt in x.index:
   z=pd.concat([x.loc[dt],ff.loc[dt]],axis=1).dropna()
   if len(z)>=8: vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  q=pd.Series(vv).dropna(); print(' decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252/h),4))
