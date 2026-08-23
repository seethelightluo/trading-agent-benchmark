import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=P.pct_change()
# Contrarian 10-day return, volatility scaled, active only in above-median cross-asset dispersion, lagged one session.
raw=-P.pct_change(10)/(r.rolling(20).std()*np.sqrt(252))
disp=r.std(axis=1).rolling(20).rank(pct=True)
f=raw.where(disp>0.5).shift(1)
f.to_csv('scripts/miner_3_20350215_dispersion_scaled_reversal_10d_signal.csv',index_label='date')
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
fr=P.pct_change(10).shift(-10); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append((dt,c))
q=pd.DataFrame(vals,columns=['date','ic']).set_index('date')
for a,b in [('2020-01-01','2027-12-31'),('2028-01-01','2031-12-31'),('2032-01-01','2035-02-13')]: print('regime',a,b,'dates',len(q.loc[a:b]),'IC',q.loc[a:b].ic.mean())
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
