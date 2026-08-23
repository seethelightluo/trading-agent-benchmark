import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-03')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); lag=r.shift(1); breadth=(lag>0).mean(axis=1); stress=(breadth<.4).astype(float)
sig=-px.pct_change(2).shift(1).mul(1+1.0*stress,axis=0); fwd=px.shift(-1)/px-1

def calc(mask=None):
 a=[];ns=[]
 for i,d in enumerate(sig.index):
  if mask is not None and not bool(mask[i]):continue
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q):a.append(q);ns.append(len(g))
 a=np.array(a);return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4)
print('end',px.index.max().date(),'overall',calc());y=sig.index.year
for q,m in [('2020-22',(y>=2020)&(y<=2022)),('2023-25',(y>=2023)&(y<=2025)),('2026',y==2026),('2027',y==2027),('last180',sig.index>=END-pd.Timedelta(days=180))]:print(q,calc(m))
print('coverage',int(sig.notna().sum().sum()),'of',sig.size,'dates',sig.index.size)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20271104_breadth_2d_reversal_signal.csv',index=False)
