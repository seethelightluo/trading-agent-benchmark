import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-11-22')
px={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 px[a]=d.loc[d.index<=E,'close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(fill_method=None)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].astype(float).reindex(p.index).ffill()
x=v.pct_change(); beta=r.rolling(60,min_periods=40).cov(x).div(x.rolling(60,min_periods=40).var(),axis=0)
resid=r.rolling(20,min_periods=15).sum()-beta.mul(x.rolling(20,min_periods=15).sum(),axis=0)
idvol=(r-beta.mul(x,axis=0)).rolling(20,min_periods=15).std()
z=-(x.rolling(5,min_periods=4).sum()); scale=z.abs().rolling(60,min_periods=40).median()+1e-8
mult=(1.0+0.75*np.tanh(z/scale)).clip(.25,1.75)
f=-resid.div(idvol+1e-8).mul(mult,axis=0)
print('continuous VIX residual recovery cutoff',E.date(),'rows',len(p),'assets',len(A))
for h in [1,5,10,20]:
 q=[];ns=[];fr=p.shift(-h)/p-1
 for t in f.index:
  zz=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(zz)>=8:q.append(spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic);ns.append(len(zz))
 q=np.array(q);print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1),np.mean(q>0)))
fr=p.shift(-10)/p-1;q=[];ds=[]
for t in f.index:
 zz=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(zz)>=8:q.append(spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic);ds.append(t)
q=np.array(q);ds=pd.Series(ds)
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-11-22'),('2033','2034-11-22')]:
 m=(ds>=lo)&(ds<=hi); y=q[m];print('regime',lo,hi,'dates',len(y),'IC %.6f ICIR %.6f'%(y.mean(),y.mean()/y.std(ddof=1)))
print('LIBRARY_CORR_EVIDENCE missing: exact common-cell reconstruction required before admission')
