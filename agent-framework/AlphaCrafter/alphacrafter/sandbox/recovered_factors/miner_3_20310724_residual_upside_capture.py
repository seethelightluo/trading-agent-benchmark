import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
files=glob.glob('../persistent/stock_data/*.csv')
px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}).sort_index().ffill()
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px=px[[a for a in A if a in px]]
r=px.pct_change();m=r.mean(axis=1);vol=r.rolling(20,min_periods=15).std()
raw=r.sub(m,axis=0).where(m>0).rolling(60,min_periods=12).mean()/vol
iv=-vol
# cross-sectional residual of upside capture on inverse volatility and momentum, lagged signal is already through t
mom=px.pct_change(20)
def resid(y, xs):
 out=y.copy()*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(xs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:].to_numpy()]; b=np.linalg.lstsq(X,z.y.to_numpy(),rcond=None)[0]
   out.loc[d,z.index]=z.y.to_numpy()-X@b
 return out
f=resid(raw,[iv,mom])
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1;z=[];ns=[];ds=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=pd.Series(z,index=ds);print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'latest120',z.tail(120).mean(),z.tail(120).mean()/z.tail(120).std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
for n,s in {'raw':raw,'invvol':iv,'mom20':mom}.items():
 q=[]
 for d in f.index:
  ok=f.loc[d].notna()&s.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],s.loc[d][ok]).statistic)
 print('corr',n,np.nanmean(q),np.nanmax(np.abs(q)))
for lab,mask in [('2020-23',f.index<'2024'),('2024-27',(f.index>='2024')&(f.index<'2028')),('2028-30',(f.index>='2028')&(f.index<'2031')),('2031',f.index>='2031')]:
 fw=px.shift(-10)/px-1; q=[]
 for d in f.index[mask]:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic)
 print(lab,len(q),np.mean(q),np.mean(q)/np.std(q,ddof=1))
