import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<120: d=get_index_daily_data(s,4000)
 if d is not None and len(d)>=120: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); m=r.mean(axis=1)
# Residual 20-day performance after removing rolling market beta exposure, lagged one completed bar.
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(m,axis=0),axis=0)
raw=res.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
f=(raw/vol.replace(0,np.nan)).shift(1)
print('assets',len(D),'rows',len(p),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q); ns.append(len(z)); dates.append(dt)
 a=np.array(a); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 # regime split by calendar half for robustness
 for label,mask in [('early',np.array([x.year<=2023 for x in dates])),('late',np.array([x.year>=2026 for x in dates]))]:
  q=a[mask]; print(label,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281228_residual_momentum_signal.csv',index=False)
