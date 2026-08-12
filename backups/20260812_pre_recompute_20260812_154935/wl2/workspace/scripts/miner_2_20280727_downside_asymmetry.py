import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  q=d.copy(); q.date=pd.to_datetime(q.date); P[s]=q.set_index('date').close.sort_index()
px=pd.concat(P,axis=1).sort_index().ffill(); r=px.pct_change()
# Downside-asymmetry trend: medium-term compounded return divided by downside deviation;
# requires persistent upside with limited negative shocks, and is lagged one completed bar.
ret=np.log1p(r.clip(lower=-.99)).rolling(40,min_periods=30).sum()
down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=30).mean())*np.sqrt(252)
up=np.sqrt((r.clip(lower=0)**2).rolling(40,min_periods=30).mean())*np.sqrt(252)
f=(ret/(down+0.25*up)).replace([np.inf,-np.inf],np.nan).shift(1)
print('universe',len(U),'loaded',len(P),'dates',len(px))
rows=[]
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   z=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(z): vals.append(z);ns.append(len(a));rows.append((dt,h,z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2028)]:
 x=pd.Series([z for d,h,z in rows if h==10 and lo<=d.year<=hi]);print('regime',lo,hi,'n',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280727_downside_asymmetry_signal.csv',index=False)
