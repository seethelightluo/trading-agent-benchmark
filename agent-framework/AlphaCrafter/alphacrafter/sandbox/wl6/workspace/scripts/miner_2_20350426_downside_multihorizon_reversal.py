import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index()
r=np.log(P).diff()
def ds(n):
 x=r.rolling(n); dn=r.where(r<0,0).pow(2).mean().pow(.5)*np.sqrt(252)
 return (r.rolling(n).sum()/dn).replace([np.inf,-np.inf],np.nan)
# Sign inversion of the previously tested continuation blend: cheap/weak assets tend to mean-revert.
f=(-(ds(60).rank(axis=1,pct=True)+ds(120).rank(axis=1,pct=True))/2).shift(1)
f.to_csv('scripts/miner_2_20350426_downside_multihorizon_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; counts=[]; turns=[]; prev=None; regimes={'early':[],'mid':[],'recent':[]}
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); counts.append(len(z)); rr=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=rr.index.intersection(prev.index); turns.append(np.mean(abs(rr[ix]-prev[ix])))
    prev=rr
    y=int(pd.Timestamp(dt).year)
    regimes['early' if y<=2027 else ('mid' if y<=2031 else 'recent')].append(c)
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(counts):.3f} coverage={np.mean(counts)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
 print(' regimes',*(f'{k}={np.mean(v):.8f}' for k,v in regimes.items()))
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum().sum()/(len(f)*15))
