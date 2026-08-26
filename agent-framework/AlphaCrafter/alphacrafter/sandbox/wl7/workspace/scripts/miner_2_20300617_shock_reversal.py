import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 try:d=get_stock_daily_data(s,days=4000)
 except Exception:d=None
 if d is None or len(d)<150:
  try:d=get_index_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None and len(d): frames[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(frames).sort_index().ffill(); ret=px.pct_change(); v5=ret.rolling(5).std(); v60=ret.rolling(60).std(); shock=v5/(v60+1e-12)
f=(-ret.rolling(5).sum()/(v5*np.sqrt(5)+1e-12)).where(shock>1.35,0).shift(1)
fy=px.shift(-10)/px-1; rows=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fy.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for lab,rr in [('full',res),('early',res.iloc[:len(res)//3]),('mid',res.iloc[len(res)//3:2*len(res)//3]),('late',res.iloc[2*len(res)//3:])]:
 m=rr.ic.mean(); sd=rr.ic.std(ddof=1); print(lab,'dates',len(rr),'avg_n',rr.n.mean(),'IC',m,'ICIR_daily',m/sd if sd else np.nan,'ICIR_ann',m/sd*np.sqrt(252) if sd else np.nan,'hit',(rr.ic>0).mean())
for h in [1,5,20,40]:
 yy=px.shift(-h)/px-1; q=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q),'dates',len(q))
print('assets',len(frames),'dates',len(px),'coverage',f.notna().sum().sum()/(f.shape[0]*15))
f.to_csv('scripts/miner_2_20300617_shock_reversal_signal.csv')
res.to_csv('scripts/miner_2_20300617_shock_reversal_ic.csv')
