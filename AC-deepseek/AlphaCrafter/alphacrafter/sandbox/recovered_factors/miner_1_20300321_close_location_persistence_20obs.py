# Single-idea validation: close-location persistence
# High signal denotes repeated closes near the daily high, normalized by each day's range.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
panels={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 panels[a]=d
close=pd.DataFrame({a:panels[a]['close'].replace(0,np.nan) for a in ASSETS}).sort_index()
sig=pd.DataFrame(index=close.index,columns=ASSETS,dtype=float)
for a,d in panels.items():
 for col in ['open','high','low','close']:
  d[col]=pd.to_numeric(d[col],errors='coerce')
 rng=(d.high-d.low).replace(0,np.nan)
 # within-day signed close location, then mean across the last 20 native observations
 clv=(2*d.close-d.high-d.low)/rng
 sig[a]=clv.rolling(20,min_periods=16).mean().reindex(sig.index)
print('candidate=close_location_persistence_20obs cutoff=',close.dropna(how='all').index.max().date())
print('signal cells',int(sig.notna().sum().sum()),'of',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,4))
for h in [1,5,10,20]:
 fwd=close.apply(lambda x:x.shift(-h)/x-1); vals=[];counts=[];dates=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z);counts.append(len(q));dates.append(dt)
 v=np.array(vals); print('h',h,'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),4),'dates',len(v),'mean_n',round(np.mean(counts),2),'min_n',min(counts))
 for label,lo,hi in [('2020-2021','2020-01-01','2021-12-31'),('2022-2023','2022-01-01','2023-12-31'),('2024-2025','2024-01-01','2025-12-31'),('2026-2030','2026-01-01','2030-12-31')]:
  z=np.array([x for x,d in zip(vals,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
  if len(z)>1: print(' ',label,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
turn=[]
for i in range(1,len(sig)):
 q=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(np.abs(q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).mean())
print('rank_turnover',round(float(np.mean(turn)),6),'adjacent_dates',len(turn))
sig.to_pickle('scripts/miner_1_close_location_persistence_candidate_signal.pkl')
