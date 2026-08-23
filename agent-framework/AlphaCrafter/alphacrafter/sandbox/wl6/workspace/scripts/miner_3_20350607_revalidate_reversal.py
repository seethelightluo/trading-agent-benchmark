import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Lagged 5D reversal scaled by trailing 20D risk, relative to cross-sectional median.
raw=P.pct_change(5); med=raw.median(axis=1); vol=r.rolling(20,min_periods=15).std()
f=(-(raw.sub(med,axis=0))/(vol+1e-8)).shift(1)
f.to_csv('scripts/miner_3_20350607_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; ns=[]; prev=None; turns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); ns.append(len(z)); dates.append(dt)
    rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     common=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[common]-prev[common])))
    prev=rr
 q=np.array(vals); recent=q[-500:] if len(q)>500 else q
 print(f'{h}D all_dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
 print(f'{h}D recent_dates={len(recent)} IC={recent.mean():.8f} ICIR={recent.mean()/recent.std(ddof=1)*np.sqrt(len(recent)):.8f} hit={np.mean(recent>0):.4f}')
print('range',P.index.min(),P.index.max(),'assets',P.shape[1])
