import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<120:d=get_index_daily_data(s,2600)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index(); r=px.pct_change(5); resid=r.sub(r.median(axis=1),axis=0); v=px.pct_change().rolling(60,min_periods=40).std()
# smooth residual component (current and prior completed observations), then lag one decision day
sig=-(.75*resid.div(v)+.25*resid.shift(1).div(v.shift(1))).shift(1)
fwd=px.shift(-5).div(px)-1
R=[]
for t in sig.index:
 z=pd.concat([sig.loc[t],fwd.loc[t]],axis=1).dropna()
 if len(z)>=8:R.append((t,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(R,columns=['date','ic','n']).set_index('date')
for lab,q in [('full',R),('2028+',R[R.index>='2028-01-01']),('2029+',R[R.index>='2029-01-01'])]:
 print(lab,'obs',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),4),'artifact dates',len(px),'instruments',len(P))
out='scripts/miner_3_20300404_residual_blend5_signal.csv';sig.to_csv(out,index_label='date');print('artifact',out)
