import pandas as pd, numpy as np
from scipy.stats import spearmanr
root='../persistent'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); return d.set_index('date').sort_index().close
dfs={s:load(root+'/stock_data/'+s+'.csv') for s in syms}; px=pd.concat(dfs,axis=1,sort=False).sort_index().loc[:'2026-07-15']; mr=load(root+'/index_data/DXY.csv').reindex(px.index).pct_change(); ret=px.pct_change()
beta=ret.rolling(40,min_periods=25).cov(mr).div(mr.rolling(40,min_periods=25).var(),axis=0); f=-beta
for h in [1,5,10]:
 fw=px.shift(-h)/px-1; vals=[];ns=[];dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 a=pd.Series(vals,index=dates); print(h,'N',len(a),'meanN',np.mean(ns),'coverage',sum(ns)/(len(a)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1: print('years',a.groupby(a.index.year).mean().round(4).to_dict())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
