import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];B='../persistent/stock_data'
def p(a):
 x=pd.read_csv(f'{B}/{a}.csv');x.date=pd.to_datetime(x.date);return x.set_index('date').close.astype(float)
px=pd.concat({a:p(a) for a in A},axis=1).sort_index().ffill().loc[:'2026-07-15'];r=px.pct_change()
# 5-day return risk adjusted by 20d realized vol, completed-day only
f=px.pct_change(5)/r.rolling(20,min_periods=15).std()
for h in [1,5,10]:
 y=px.shift(-h)/px-1; zics=[];ns=[]
 for d in f.index:
  z=f.loc[d];q=y.loc[d];ok=z.notna()&q.notna()
  if ok.sum()>=8:zics.append(spearmanr(z[ok],q[ok]).statistic);ns.append(ok.sum())
 a=np.array(zics); print(h,len(a),round(np.mean(ns),2),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),round(np.mean(a>0),4))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4))
for name,v in {'mom20':px.pct_change(20),'rev5':-px.pct_change(5),'mom60':px.pct_change(60)}.items():
 q=pd.concat([f.stack().rename('f'),v.stack().rename('x')],axis=1).dropna();print('corr',name,round(q.corr(method='spearman').iloc[0,1],4))
print('period',f.index.min().date(),f.index.max().date())
