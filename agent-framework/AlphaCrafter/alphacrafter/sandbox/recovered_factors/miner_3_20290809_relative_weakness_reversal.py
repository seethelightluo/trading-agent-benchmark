import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',header=None,names=['date','open','close','high','low','volume','change','pct']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index().apply(pd.to_numeric,errors='coerce'); r=p.pct_change()
# contrarian relative weakness, with medium trend control; all rolling inputs are lagged one day
r20=p.pct_change(20); r60=p.pct_change(60); vol=r.rolling(20).std()
cs20=r20.sub(r20.mean(axis=1),axis=0); cs60=r60.sub(r60.mean(axis=1),axis=0)
sig=(-(cs20) + 0.30*cs60).div(vol)
sig=sig.shift(1)
print('cutoff',p.index.max(),'dates',len(p),'assets',len(assets))
for h in [1,5,10,20]:
  f=p.pct_change(h).shift(-h)
  vals=[]; turnovers=[]; nvalid=[]
  for d in sig.index:
    x=sig.loc[d]; y=f.loc[d]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
      vals.append(spearmanr(x[ok],y[ok]).statistic); nvalid.append(ok.sum())
  z=np.asarray(vals); print(h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'dates',len(z),'mean_n',np.mean(nvalid))
# 10d rank turnover
ranks=sig.rank(axis=1,pct=True); q=ranks.diff(10).abs().mean(axis=1).dropna(); print('turnover_proxy',q.mean(),'coverage',sig.notna().sum().sum()/(sig.size))
