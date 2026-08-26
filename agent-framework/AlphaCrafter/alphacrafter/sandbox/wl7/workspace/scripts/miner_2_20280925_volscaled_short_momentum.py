import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in u:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=3000)
  except: pass
  if d is not None and len(d): break
 if d is not None and len(d):
  x=d[['date','close']].dropna(); x.date=pd.to_datetime(x.date); fs[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(fs).sort_index().ffill(); ret=p.pct_change();
# Five-session lagged momentum divided by lagged 20-session realized volatility; no look-ahead.
sig=((p.shift(2)/p.shift(7)-1)/(ret.rolling(20).std()*np.sqrt(252)).shift(2)).shift(1)
out=[]
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q);dates.append(dt);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 if h==10:
  rk=sig.rank(axis=1,pct=True); t=[]
  for i in range(1,len(rk)):
   c=rk.iloc[i].dropna().index.intersection(rk.iloc[i-1].dropna().index)
   if len(c)>=8:t.append(abs(rk.iloc[i][c]-rk.iloc[i-1][c]).mean())
  print('TURN',round(np.mean(t),6),'COVERAGE',round(sig.notna().sum().sum()/(sig.shape[0]*len(u)),4))
  for lab,a0,b in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31')]:
   q=np.array([vals[i] for i,d in enumerate(dates) if pd.Timestamp(a0)<=d<=pd.Timestamp(b)]); print('REG',lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6) if len(q)>1 else None)
sig.index.name='date';sig.to_csv('scripts/miner_2_20280925_volscaled_short_momentum_signal.csv'); print('range',p.index.min(),p.index.max(),'assets',len(fs))
