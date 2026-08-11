import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<120: d=get_index_daily_data(s,4000)
 if d is not None and len(d)>=120: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Directional path-efficiency trend: net 30d log return divided by total absolute path, lagged one bar.
net=r.rolling(30,min_periods=20).sum(); path=r.abs().rolling(30,min_periods=20).sum()
f=(net/(path+1e-12)).shift(1)
print('rows',len(p),'assets',len(D),'range',p.index.min(),p.index.max(),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q); ns.append(len(z))
 a=np.array(a); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
y=np.log(p).shift(-10)-np.log(p)
for label,lo in [('2027+','2027-01-01'),('2028YTD','2028-01-01'),('2026','2026-01-01')]:
 a=[]
 for dt in f.loc[lo:].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q)
 a=np.array(a); print(label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281116_path_efficiency_signal.csv',index=False)
