import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Candidate: reversal of 20D relative strength, conditioned on broad market trend.
# When cross-asset breadth is weak, emphasize mean reversion; when broad trend is strong,
# damp reversal. Breadth is lagged and used only as a smooth multiplier.
breadth=(r.rolling(20).mean()>0).mean(axis=1)
relative=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0)
calm=r.rolling(40).std().replace(0,np.nan)*np.sqrt(252)
# smooth state: multiplier ranges [0.5,1.5], with stronger reversal under weak breadth
mult=(1.5-breadth).clip(0.5,1.5)
f=(-relative/calm).mul(mult,axis=0).shift(1)
f.to_csv('scripts/miner_3_20350510_breadth_conditioned_calm_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; counts=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); counts.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[ix]-prev[ix])))
    prev=rr
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(counts):.3f} coverage={np.mean(counts)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
# regime split for admission horizon
fr=P.pct_change(10).shift(-10); vals=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c); dates.append(dt)
q=pd.Series(vals,index=pd.to_datetime(dates)); print('regime_10d',q.loc[q.index<'2028'].mean(),q.loc[(q.index>='2028')&(q.index<'2032')].mean(),q.loc[q.index>='2032'].mean())
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
