import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
 D[s]=d
px=pd.DataFrame(D).sort_index(); px=px.loc[:'2029-05-02']
r=px.pct_change()
# directional efficiency: net return divided by path length, then volatility-scaled
net=px.pct_change(20)
path=r.abs().rolling(20).sum()
eff=net/path
# require trend persistence through positive-day breadth, interpretable composite
bread=(r>0).rolling(20).mean()
f=eff*(0.5+ bread)
for h in [1,5,10,20]:
 vals=[]; turns=[]; ns=[]
 for i in range(20,len(px)-h):
  dt=px.index[i]; nxt=px.iloc[i+h]/px.iloc[i]-1
  x=f.iloc[i]
  z=pd.concat([x,nxt],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 # rank turnover sampled daily
 print(h,len(vals),np.mean(ns),np.nanmean(vals),np.nanstd(vals,ddof=1),np.nanmean(vals)/np.nanstd(vals,ddof=1))
# signal coverage and rank turnover 10d
valid=f.notna().sum(axis=1); print('coverage',valid.mean()/15,'dates',len(px))
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
