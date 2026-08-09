import pandas as pd, numpy as np
from scipy.stats import spearmanr
root='../persistent'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv(root+'/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in syms},axis=1).sort_index().loc[:'2026-07-15']; r=px.pct_change()
# Upside/downside asymmetry: recent upside average relative to downside magnitude, stabilized
up=r.clip(lower=0).rolling(40,min_periods=25).mean(); dn=(-r.clip(upper=0)).rolling(40,min_periods=25).mean()
f=(up-dn)/(up+dn+1e-12) # higher persistent upside skew
print('dates',px.index.min(),px.index.max(),'assets',px.shape[1])
for h in [1,5,10]:
 fw=px.shift(-h)/px-1; vals=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=pd.Series(vals,index=ds); print(h,'N',len(a),'meanN',round(np.mean(ns),2),'coverage',round(sum(ns)/(len(a)*15),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 print('yr',a.groupby(a.index.year).mean().round(4).to_dict())
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
# correlation with existing simple factors diagnostic
for name,x in [('rev',-r.rolling(5).sum()),('mom',r.rolling(20).sum()/r.rolling(20).std())]:
 print('corr',name,f.stack().corr(x.stack()))
