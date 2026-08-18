import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0: d=get_index_daily_data(s,3000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index().ffill()
# dual-horizon trend agreement: medium momentum, only retain when long trend agrees; lag one day
r20=P/P.shift(20)-1; r60=P/P.shift(60)-1
vol=P.pct_change().rolling(20).std()*np.sqrt(252)
sig=(r20/vol.replace(0,np.nan))*np.sign(r60)
sig=sig.shift(1)
fwd=P.shift(-10)/P-1
rows=[]; dates=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
ic=np.array(rows,float); mean= np.nanmean(ic); sd=np.nanstd(ic,ddof=1)
print('dates',len(ic),'avgN',np.nanmean([pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates]),'coverage',len(ic)/len(sig.index),'IC10',mean,'ICIR',mean/sd,'hit',np.mean(ic>0))
for h in [5,10,20,40]:
 q=P.shift(-h)/P-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
# turnover rank signal
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('turnover',turn)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 m=[(d, v) for d,v in zip(dates,ic) if a<=str(d.year)<=b]
 print('regime',a,b,'n',len(m),'icir',np.mean([v for _,v in m])/np.std([v for _,v in m],ddof=1) if len(m)>1 else np.nan,'ic',np.mean([v for _,v in m]) if m else np.nan)
out=pd.DataFrame({'date':dates,**{s:sig.loc[dates,s].values for s in sig.columns}})
out.to_csv('scripts/miner_3_20340707_dual_horizon_agreement_signal.csv',index=False)
