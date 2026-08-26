import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
    try:
        x=get_stock_daily_data(a,days=2200)
        if x is not None and len(x): D[a]=x.set_index('date')['close'].astype(float)
    except Exception as e: print('ERR',a,e)
p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change()
# Drawdown-recovery: trailing 20d return divided by magnitude of worst close-to-close drawdown in trailing 60d.
wealth=p/p.shift(20)
rollmax=p.rolling(60,min_periods=40).max(); dd=p/rollmax-1
worst=-dd.rolling(60,min_periods=40).min()
f=wealth-1
f=f/worst.replace(0,np.nan)
print('dates',p.index.min(),p.index.max(),'assets',len(D))
for h in [1,5,10,20]:
    vals=[]; dates=[]; ns=[]
    for i in range(len(p)-h):
        q=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
        if len(q)>=8:
            vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); dates.append(p.index[i]); ns.append(len(q))
    s=pd.Series(vals,index=dates).dropna()
    print('h',h,'n',len(s),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
    for label,mask in [('recent252',np.arange(len(s))>=max(0,len(s)-252)),('online2026',s.index>=pd.Timestamp('2026-07-16')),('2027',s.index>=pd.Timestamp('2027-01-01'))]:
        z=s[mask]
        if len(z): print(label,len(z),z.mean(),z.mean()/z.std(),(z>0).mean())
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
