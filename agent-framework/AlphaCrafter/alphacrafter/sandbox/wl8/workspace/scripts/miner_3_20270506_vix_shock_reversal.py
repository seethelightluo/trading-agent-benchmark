import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-05-05')
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date')
# lagged VIX shock: yesterday's one-day change relative to its trailing realized scale
v['chg']=v.close.pct_change(); v['scale']=v.chg.shift(1).rolling(20,min_periods=15).std(); v['shock']=(v.chg.shift(1)/v.scale).clip(-3,3)
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date'); x=x.merge(v[['date','shock']],on='date',how='left')
 # reversal amplified only by magnitude of lagged VIX shock, avoiding VIX level overlap
 x['sig']=-(x.close/x.open-1)*(1+0.50*x.shock.abs().clip(0,2)); x['fwd']=x.close.shift(-1)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','sig','fwd']])
z=pd.concat(rows).dropna(); obs=[]; ns=[]
for d,g in z.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: obs.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g))
a=np.array(obs); print('dates',len(a),'rows',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(len(z)/(15*z.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026+',z.date>=pd.Timestamp('2026-01-01')),('2027',z.date.dt.year==2027)]:
 q=[]
 for _,g in z[cut].groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:q.append(spearmanr(g.sig,g.fwd).statistic)
 q=np.array(q); print(label,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
z[['date','symbol','sig']].to_csv('scripts/miner_3_20270506_vix_shock_reversal_signal.csv',index=False)
