import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# residual momentum: lagged 60d asset return minus cross-sectional median 60d return
mom=p.pct_change(60).shift(1); f=mom.sub(mom.median(axis=1),axis=0)
rows=[]
for h in [1,5,10]:
 ic=[]; n=[]
 fr=p.pct_change(h).shift(-h)
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
 a=np.array(ic); a=a[np.isfinite(a)]
 print('H',h,'dates',len(a),'avgN',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
# coverage and rank turnover
print('matrix coverage',round(f.notna().mean().mean(),4),'dates',len(f),'assets',len(px))
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean().mean(),4))
# yearly daily
fr=p.pct_change(1).shift(-1)
for yr,g in f.groupby(f.index.year):
 a=[]
 for dt in g.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);a=a[np.isfinite(a)]
 print('Y',yr,'n',len(a),'ic',round(a.mean(),5) if len(a) else None,'icir',round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)
# artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_residual_momentum60.csv',index=False)
