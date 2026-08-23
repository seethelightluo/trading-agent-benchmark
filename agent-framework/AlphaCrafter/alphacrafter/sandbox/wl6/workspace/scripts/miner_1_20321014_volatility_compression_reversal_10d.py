import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-13']; ret=p.pct_change()
# Compression followed by reversal: recent 5d return, scaled by prior 20d volatility and gated by low recent/prior volatility ratio.
vol5=ret.rolling(5).std(); vol20=ret.rolling(20).std(); compression=(vol5/vol20).replace(0,np.nan)
sig=(-ret.rolling(5).sum()/vol20).where(compression<0.80)
fwd=p.shift(-10).div(p)-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=volatility-compression 5d reversal, horizon=10d')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',sig.notna().sum().sum()/sig.size)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10,20,40]:
 fw=p.shift(-h).div(p)-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('decay',h,'n',len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)))
r['year']=r.index.year; print('year_IC');print(r.groupby('year').ic.mean().to_string())
