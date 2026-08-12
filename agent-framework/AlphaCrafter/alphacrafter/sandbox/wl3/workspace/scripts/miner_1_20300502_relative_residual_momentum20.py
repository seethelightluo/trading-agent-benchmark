import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index(); lr=np.log(px).diff()
# Relative strength residual: asset's 20-session log return versus same-date cross-sectional median.
r20=lr.rolling(20,min_periods=20).sum(); med=r20.median(axis=1); sig=(r20.sub(med,axis=0)).shift(1)
# damp extremes by 60d volatility, preserving interpretable relative momentum
vol=lr.rolling(60,min_periods=40).std()*np.sqrt(60)
sig=(sig/vol.shift(1)).replace([np.inf,-np.inf],np.nan)
def calc(f):
 R=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t].rename('signal'),f.loc[t].rename('fwd')],axis=1).dropna()
  if len(z)>=8:R.append((t,spearmanr(z.signal,z.fwd).statistic,len(z)))
 return pd.DataFrame(R,columns=['date','ic','n']).set_index('date')
def stats(q): return (q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
R=calc(px.shift(-1).div(px)-1)
print('dates',len(px),'instruments',len(P),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(sig.notna().sum(axis=1).mean()/len(U),4))
for lab,q in [('full',R),('2020-22',R.loc['2020':'2022']),('2023-25',R.loc['2023':'2025']),('2026-27',R.loc['2026':'2027']),('2028+',R.loc['2028':]),('recent250',R.tail(250))]:
 if len(q)>2: print(lab,'IC',round(stats(q)[0],6),'ICIR',round(stats(q)[1],6),'hit',round(stats(q)[2],4),'n',len(q))
for h in [3,5,10]:
 q=calc(px.shift(-h).div(px)-1); print('decay',h,'IC',round(stats(q)[0],6),'ICIR',round(stats(q)[1],6),'n',len(q))
rank=sig.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
out='scripts/miner_1_20300502_relative_residual_momentum20_signal.csv';sig.to_csv(out,index_label='date');print('artifact',out)
