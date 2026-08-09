import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index();r=p.pct_change()
base=-(p/p.shift(10)-1)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)); trend=(p/p.shift(20)-1)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)); rev5=-(p/p.shift(5)-1)/(r.rolling(5,min_periods=4).std()*np.sqrt(5))
def resid(y,X):
 out=y* np.nan
 for d in y.index:
  q=pd.concat([y.loc[d].rename('y'),X.loc[d]],axis=1).dropna()
  if len(q)>=8:
   Z=np.column_stack([np.ones(len(q)),q.iloc[:,1:].values]); b=np.linalg.lstsq(Z,q.y.values,rcond=None)[0];out.loc[d,q.index]=q.y.values-Z@b
 return out
# residualized cross-section against trend and 5d reversal; lag after residual construction
f=resid(base,pd.DataFrame({'trend':trend.stack(),'rev5':rev5.stack()}).unstack()).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;z=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('H',h,'dates',len(z),'N',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().stack().mean(),4),'turn10',round((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().stack().dropna().mean(),4))
# exact max against library proxies
mx=0;who=''
for k,x in {'trend':trend,'rev5':rev5,'raw10':-(p/p.shift(10)-1),'short':-(p/p.shift(5)-1)}.items():
 q=pd.concat([f.stack().rename('f'),x.shift(1).stack().rename('x')],axis=1).dropna();rho=spearmanr(q.f,q.x).statistic;print('CORR',k,round(rho,5));
 if abs(rho)>mx:mx=abs(rho);who=k
print('MAXCORR',who,round(mx,5))