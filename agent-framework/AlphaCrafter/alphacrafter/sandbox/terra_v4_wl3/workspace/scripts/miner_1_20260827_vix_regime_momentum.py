import pandas as pd,numpy as np
from scipy.stats import spearmanr
base='../persistent'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(f'{base}/stock_data/{s}.csv',parse_dates=['date']).set_index('date').close for s in syms}
vix=pd.read_csv(f'{base}/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.pct_change()
r=pd.DataFrame(px).pct_change(); vr=vix.reindex(r.index)
vlevel=vr.rolling(60,min_periods=40).mean() # regime level is rolling VIX return, no future
# use stress sign: momentum is reduced after sustained positive VIX-return regime
scale=(1-8*vlevel.clip(-.03,.03)).rename('scale')
factor=r.rolling(20,min_periods=20).sum().mul(scale,axis=0).loc[:'2026-07-15']; r=r.loc[:'2026-07-15']
def calc(h):
 vals=[]
 for i,d in enumerate(factor.index):
  if i+h>=len(r): continue
  z=pd.concat([factor.loc[d].rename('x'),r.iloc[i+1:i+h+1].sum().rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic)
 a=np.array(vals); return len(a),round(float(a.mean()),5),round(float(a.mean()/a.std(ddof=1)),5),round(float((a>0).mean()),4)
for h in [1,5,10]: print(h,calc(h))
valid=factor.notna().sum(axis=1); print('dates',sum(valid>=8),'avg names',round(valid[valid>=8].mean(),2),'coverage',round(factor.notna().sum().sum()/factor.size,4))
for yr in range(2020,2027):
 vals=[]
 for d in factor.loc[str(yr)].index:
  i=r.index.get_loc(d)
  if i+1>=len(r):continue
  z=pd.concat([factor.loc[d].rename('x'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic)
 a=np.array(vals); print(yr,len(a),round(a.mean(),4),round(a.mean()/a.std(ddof=1),4))
