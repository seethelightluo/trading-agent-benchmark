import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Defensive trend-quality: medium trend, persistence of up days, and inverse downside risk.
trend=P.pct_change(30)
persistence=(r>0).rolling(30,min_periods=20).mean()
downside=(-r.clip(upper=0)).rolling(30,min_periods=20).std()
vol=r.rolling(30,min_periods=20).std()
quality=(0.5+ persistence).clip(lower=0.2)
f=(trend*quality/(0.5*vol+0.5*downside)).shift(1)
f.to_csv('scripts/miner_3_20350524_defensive_trend_quality_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; counts=[]; prev=None; turns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); counts.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     common=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[common]-prev[common])))
    prev=rr
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(counts):.3f} coverage={np.mean(counts)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
