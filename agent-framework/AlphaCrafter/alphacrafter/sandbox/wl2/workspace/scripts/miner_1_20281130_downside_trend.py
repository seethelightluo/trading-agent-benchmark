import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3200)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Downside-risk-adjusted medium trend: 30d log momentum / downside deviation,
# with cross-sectional demeaning and lagged 5d breadth gate. Downside risk is
# more relevant to long-only allocation than total volatility.
down=r.clip(upper=0).rolling(30,min_periods=20).std()*np.sqrt(30)
raw=np.log(p).diff(30)/(down+1e-12)
breadth=(np.log(p).diff(30)>0).mean(axis=1).rolling(5,min_periods=3).mean()
gate=(2*(breadth-.5)).clip(-1,1)
f=raw.sub(raw.median(axis=1),axis=0).mul(gate,axis=0).shift(1)
print('rows',len(p),'assets',len(D),'coverage',round(f.notna().mean().mean(),4))
for h in [5,10,20,30]:
 y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,mask in [('2020-22', (f.index>='2020-01-01')&(f.index<'2023-01-01')),('2023-25',(f.index>='2023-01-01')&(f.index<'2026-01-01')),('2026+',f.index>='2026-01-01'),('2028YTD',f.index>='2028-01-01'),('recent',f.index>='2028-05-01')]:
 y=np.log(p).shift(-20)-np.log(p); a=[]
 for dt in f.index[mask]:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q)
 a=np.array(a); print(label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
f.to_csv('scripts/miner_1_20281130_downside_trend_signal.csv')
