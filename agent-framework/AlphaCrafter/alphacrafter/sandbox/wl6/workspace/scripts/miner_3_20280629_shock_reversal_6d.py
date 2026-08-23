import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 return d
raw={s:fetch(s) for s in U}; close=pd.DataFrame({s:d.set_index('date')['close'] for s,d in raw.items() if d is not None}).sort_index(); ret=close.pct_change()
# Downside deviation treats non-loss observations as zero, preserving broad coverage.
down=ret.clip(upper=0); dn10=down.rolling(10,min_periods=7).std(); dn40=down.rolling(40,min_periods=25).std(); scale=(dn40/dn10).clip(.5,2.0)
factor=-(ret.rolling(6,min_periods=6).sum()*scale).shift(1)
for h in [1,5,10]:
 vals=[]
 for dt in factor.index:
  x=pd.concat([factor.loc[dt],close.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(x)>=8: vals.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); sd=z.ic.std(ddof=1)
 print(f'H={h} dates={len(z)} avgN={z.n.mean():.2f} IC={ic:.8f} ICIR={ic/sd*np.sqrt(len(z)) if sd>0 else np.nan:.8f} hit={(z.ic>0).mean():.4f}')
 if h==1:
  for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2028')]:
   q=z.loc[a:b]
   if len(q): print(f' regime {a}-{b} dates={len(q)} IC={q.ic.mean():.8f} ICIR={q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)) if q.ic.std(ddof=1)>0 else np.nan:.8f}')
valid=factor.notna().sum(axis=1); ranks=factor.rank(axis=1,pct=True); print(f'coverage={valid.mean()/len(U):.6f} rank_turnover={ranks.diff().abs().mean(axis=1).mean():.8f} dates={len(close)} instruments={close.shape[1]} last={close.index.max()}')
