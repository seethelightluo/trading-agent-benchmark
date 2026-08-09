import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path):
 d=pd.read_csv(path); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.astype(float)
root='../persistent'
px=pd.concat({s:load(root+'/stock_data/'+s+'.csv') for s in U},axis=1).sort_index().loc[:'2026-07-15']
r=px.pct_change(); dxy=load(root+'/index_data/DXY.csv').reindex(px.index).ffill().pct_change()
# Candidate: negative rolling DXY beta (cross-sectional macro sensitivity), 60d.
beta=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U:
 x=r[s]; beta[s]=(x.mul(dxy).rolling(60,min_periods=40).mean()-x.rolling(60,min_periods=40).mean()*dxy.rolling(60,min_periods=40).mean())/dxy.rolling(60,min_periods=40).var()
f=-beta
fw=px.shift(-1)/px-1
vals=[]; ns=[]; dates=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(dt)
a=pd.Series(vals,index=dates)
print('dxy_beta_negative','horizon',1,'dates',len(a),'avgN',np.mean(ns),'coverage',sum(ns)/(len(a)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('annual',a.groupby(a.index.year).mean().round(4).to_dict(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# test second distinct macro factor: trailing 20d cross-asset residual to DXY, i.e. asset return minus beta*DXY return, momentum of residual
res=r-beta.mul(dxy,axis=0); fac=res.rolling(10,min_periods=8).sum()
vals=[];ns=[]
for dt in fac.index:
 q=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
a=pd.Series(vals)
print('dxy_residual_10d','dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
