import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-05-19')
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date')
v['chg']=v.close.pct_change(); v['scale']=v.chg.shift(1).rolling(20,min_periods=15).std(); v['shock']=(v.chg.shift(1)/v.scale).clip(-2,2)
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].sort_values('date').merge(v[['date','shock']],on='date',how='left')
 # Directional macro modulation: lagged positive VIX shocks strengthen reversal; falling VIX shocks attenuate it
 x['sig']=-(x.close/x.open-1)*(1+0.50*x.shock.fillna(0)).clip(0,2)
 x['fwd1']=x.close.shift(-1)/x.close-1; x['fwd5']=x.close.shift(-5)/x.close-1; x['symbol']=s
 rows.append(x[['date','symbol','sig','fwd1','fwd5']])
z=pd.concat(rows).dropna(subset=['sig','fwd1']); obs=[]; ns=[]
for d,g in z.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fwd1.nunique()>1: obs.append(spearmanr(g.sig,g.fwd1).statistic); ns.append(len(g))
a=np.array(obs); print('dates',len(a),'rows',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(len(z)/(15*z.date.nunique()),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for h in ['fwd5']:
 oo=[]
 for d,g in z.dropna(subset=[h]).groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g[h].nunique()>1: oo.append(spearmanr(g.sig,g[h]).statistic)
 q=np.array(oo); print(h,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for label,cut in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027)]:
 q=[]
 for _,g in z[cut].groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fwd1.nunique()>1:q.append(spearmanr(g.sig,g.fwd1).statistic)
 q=np.array(q); print(label,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
z[['date','symbol','sig']].to_csv('scripts/miner_3_20270520_vix_direction_reversal_signal.csv',index=False)
