# Single-idea validation: persistent range-position extreme conditioned reversal.
# A short-term reversal signal is active in proportion to the PREVIOUS persistent
# daily close-location extreme: persistent closes at either range edge indicate
# exhaustion, so a high signal favors assets with a recent directionally adverse move.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 for x in ['open','high','low','close']: d[x]=pd.to_numeric(d[x],errors='coerce')
 p[a]=d
c=pd.DataFrame({a:p[a].close.replace(0,np.nan) for a in A}).sort_index(); r=c.pct_change(fill_method=None)
sig=pd.DataFrame(index=c.index,columns=A,dtype=float)
for a,d in p.items():
 clv=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan)
 # entirely lagged state: yesterday's 20-observation mean close location strength
 edge=clv.rolling(20,min_periods=16).mean().abs().shift(1)
 # smooth, interpretable activation; 60-observation own-history normalization
 activation=edge/edge.rolling(60,min_periods=40).median().replace(0,np.nan)
 sig[a]=(-c[a].pct_change(5)/r[a].rolling(5,min_periods=4).std()*activation).reindex(sig.index)
print('candidate=persistent_range_position_extreme_conditioned_volnorm_reversal_5_20_60obs cutoff=',c.dropna(how='all').index.max().date())
print('signal cells',int(sig.notna().sum().sum()),'of',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,4))
allvals={}
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1; vals=[];ns=[];ds=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z);ns.append(len(q));ds.append(dt)
 v=np.array(vals);allvals[h]=(v,ds)
 print('h',h,'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),4),'dates',len(v),'mean_n',round(np.mean(ns),2),'min_n',min(ns))
 for lab,lo,hi in [('2020-2021','2020-01-01','2021-12-31'),('2022-2023','2022-01-01','2023-12-31'),('2024-2025','2024-01-01','2025-12-31'),('2026-2030','2026-01-01','2030-12-31')]:
  x=np.array([z for z,dt in zip(v,ds) if pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)])
  if len(x)>1: print(' ',lab,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
t=[]
for i in range(1,len(sig)):
 q=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
 if len(q)>=8:t.append(np.abs(q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).mean())
print('rank_turnover',round(np.mean(t),6),'adjacent_dates',len(t))
sig.to_pickle('scripts/miner_1_persistent_range_extreme_reversal_candidate_signal.pkl')
