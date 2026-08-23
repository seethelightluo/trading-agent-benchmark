import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(Path('../persistent/stock_data')/f'{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().loc[:'2032-10-13']; r=p.pct_change()
# Cross-asset dispersion is observation-only conditioning; amplify reversal in high dispersion regimes.
disp=r.T.rolling(5).std().T.mean(axis=1)
reg=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
sig=-r.rolling(20).sum().mul(1+0.5*reg,axis=0)
print('candidate=dispersion-conditioned 20d reversal; cutoff=2032-10-13')
print('data_dates',len(p),'instruments',len(syms))
for h in [5,10,20,40]:
 fw=p.shift(-h).div(p)-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(rows).dropna(); print('horizon',h,'dates',len(x),'avg_n',len(syms),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('coverage %.6f turnover %.6f'%(sig.notna().sum().sum()/sig.size,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
