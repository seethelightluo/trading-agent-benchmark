import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for a in A:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv'); q.date=pd.to_datetime(q.date); d[a]=q.set_index('date').close
px=pd.DataFrame(d).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=15).std()
# 40-observation recovery, residualized against trend and range position
base=(px/px.rolling(40,min_periods=24).min()-1)/(v*np.sqrt(20))
trend=(px/px.shift(20)-1)/v
ran=(px-px.rolling(40,min_periods=24).min())/(px.rolling(40,min_periods=24).max()-px.rolling(40,min_periods=24).min())
f=pd.DataFrame(index=px.index,columns=A,dtype=float)
for dt in px.index:
 z=pd.DataFrame({'y':base.loc[dt],'t':trend.loc[dt],'q':ran.loc[dt]}).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t,z.q]; b=np.linalg.lstsq(X,z.y,rcond=None)[0]; f.loc[dt,z.index]=z.y-X@b
f=f.shift(1)
print('FACTOR residual_drawdown_recovery_40 dates',len(px),'assets',len(A),'coverage',round(f.notna().sum().sum()/f.size,4))
def run(h):
 y=px.pct_change(h).shift(-h); o=[]; ns=[]; ds=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:o.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(o,index=pd.to_datetime(ds));return s,np.mean(ns)
for h in [1,5,10,20]:
 s,n=run(h);print('H',h,'dates',len(s),'avgN',round(n,2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent60',round(s.tail(60).mean(),6))
s,_=run(1)
for y,g in s.groupby(s.index.year):print('REG',y,len(g),round(g.mean(),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
lib={'trend':trend,'reversal':-(px/px.shift(5)-1)/r.rolling(5).std(),'vol':v,'transition':r.rolling(5).std()/r.rolling(60).std(),'peer':px.pct_change(20)-px.pct_change(20).median(axis=1).values[:,None]}
co={}
for k,x in lib.items():
 z=pd.concat([f.stack(),x.shift(1).stack()],axis=1).dropna();co[k]=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);print('CORR',k,round(co[k],6))
print('MAX',round(max(abs(v) for v in co.values()),6),co)
