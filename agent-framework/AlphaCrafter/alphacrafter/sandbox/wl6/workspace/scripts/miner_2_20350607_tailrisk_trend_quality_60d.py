import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Tail-risk trend quality: medium-term return rewarded, but penalized by downside
# volatility and the average magnitude of the worst 10% daily observations.
ret=r.rolling(60,min_periods=40).sum()
down=r.where(r<0).rolling(40,min_periods=25).std()
tail=r.rolling(60,min_periods=40).apply(lambda x: np.mean(np.sort(x)[max(0,int(len(x)*.1)-1):int(len(x)*.1)+1]) if np.isfinite(x).sum()>20 else np.nan,raw=True)
den=(down.fillna(0)+tail.abs()).replace(0,np.nan)
f=(ret/den).shift(1)
f.to_csv('scripts/miner_2_20350607_tailrisk_trend_quality_60d_signal.csv',index_label='date')
for h in [5,10,20,40,60]:
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
fr=P.pct_change(60).shift(-60); vals=[];dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append(c);dates.append(dt)
q=pd.Series(vals,index=pd.to_datetime(dates))
print('regime_60d',q.loc[q.index<'2028'].mean(),q.loc[(q.index>='2028')&(q.index<'2032')].mean(),q.loc[q.index>='2032'].mean())
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
