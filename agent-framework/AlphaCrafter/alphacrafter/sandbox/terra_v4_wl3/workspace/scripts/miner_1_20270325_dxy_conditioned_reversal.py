import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); root='../persistent/stock_data'
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
px=pd.concat([load(f'{root}/{a}.csv').close.rename(a) for a in U],axis=1).sort_index().loc[:cut]
# Dollar-shock conditioned reversal: fade recent relative returns, but only amplify when lagged DXY move is unusually large.
dxy=load('../persistent/index_data/DXY.csv').close.reindex(px.index).ffill()
r=px.pct_change(); rel3=r.rolling(3,min_periods=3).sum(); rel3=rel3.sub(rel3.median(axis=1),axis=0)
shock=dxy.pct_change(5).abs(); z=(shock.shift(1)-shock.shift(1).rolling(120,min_periods=60).median())
state=(1+1.5*(z>0).astype(float)).rename('state')
fac=(-rel3.shift(1)).mul(state,axis=0)
y={h:px.pct_change(h).shift(-h) for h in [1,5,10]}
fac.to_csv('scripts/miner_1_20270325_dxy_conditioned_reversal_signal.csv')
print('assets',len(U),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in y:
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],y[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(dt); ns.append(len(q))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'dates',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(U),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
