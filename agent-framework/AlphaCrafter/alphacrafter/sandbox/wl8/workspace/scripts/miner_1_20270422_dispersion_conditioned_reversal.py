import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-04-21')
# Candidate: amplify 3d cross-asset reversal when lagged cross-sectional dispersion is elevated.
# Dispersion uses only completed same-date returns and its trailing median is lagged.
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date')
 x['r1']=x.close.pct_change(); x['r3']=x.close/x.close.shift(3)-1; x['symbol']=s
 rows.append(x[['date','symbol','r1','r3','close']])
z0=pd.concat(rows)
disp=z0.groupby('date').r1.std().rename('disp')
disp_med=disp.shift(1).rolling(60,min_periods=30).median(); regime=(disp.shift(1)/disp_med-1).clip(-0.5,1.5)
z=z0.merge(regime.rename('regime'),on='date',how='left')
z['sig']=-z.r3*(1+0.8*z.regime)
z['fwd']=z.groupby('symbol').close.shift(-1)/z.close-1
z=z.dropna(subset=['sig','fwd'])
obs=[];ns=[]
for d,g in z.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1: obs.append(spearmanr(g.sig,g.fwd).statistic); ns.append(len(g))
a=np.array(obs)
print('dates',len(a),'rows',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(len(z)/(15*z.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026+',z.date>=pd.Timestamp('2026-01-01')),('2027',z.date.dt.year==2027)]:
 q=[]
 for _,g in z[cut].groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fwd.nunique()>1:q.append(spearmanr(g.sig,g.fwd).statistic)
 q=np.array(q); print(label,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
z[['date','symbol','sig']].to_csv('scripts/miner_1_20270422_dispersion_conditioned_reversal_signal.csv',index=False)
