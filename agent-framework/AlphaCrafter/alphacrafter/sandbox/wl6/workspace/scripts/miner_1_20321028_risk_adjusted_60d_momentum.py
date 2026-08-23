import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-27']; ret=p.pct_change()
# medium-term trend strength: 60d price return scaled by trailing 30d daily volatility
sig=p.pct_change(60).div(ret.rolling(30).std().replace(0,np.nan))
rows=[]
for h in [5,10,20,40]:
 fwd=p.shift(-h).div(p)-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
 if h==10:
  print('coverage',sig.notna().sum().sum()/sig.size,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  print('annual',q.assign(year=q.index.year).groupby('year').ic.mean().to_dict())
