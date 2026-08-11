import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3000)
 if d is not None: D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Downside-adjusted recovery: medium-term return per downside deviation, with a mild short-term confirmation.
down=r.where(r<0,0).rolling(40,min_periods=25).std()
ret40=np.log(p).diff(40); ret10=np.log(p).diff(10)
f=(0.75*ret40/(down*np.sqrt(40))+0.25*ret10/(r.rolling(20,min_periods=15).std()*np.sqrt(20))).replace([np.inf,-np.inf],np.nan).shift(1)
print('universe',len(U),'columns',len(D),'factor_dates',len(f))
for h in [1,3,5,10,20]:
 y=np.log(p).shift(-h)-np.log(p); a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q); ns.append(len(z)); ds.append(dt)
 a=np.array(a)
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for name,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_25','2023-01-01','2025-12-31'),('2026_27','2026-01-01','2027-12-31'),('2028YTD','2028-01-01','2028-09-05')]:
  x=a[[d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi) for d in ds]]
  if len(x)>10: print(' ',name,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
f.to_csv('scripts/miner_1_20280907_downside_recovery_efficiency_signal.csv')
