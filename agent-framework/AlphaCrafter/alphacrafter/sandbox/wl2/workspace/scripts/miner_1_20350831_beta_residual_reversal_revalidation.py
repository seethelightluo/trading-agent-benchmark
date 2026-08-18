import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x)>100:
        z=x[['date','close']].copy(); z['date']=pd.to_datetime(z.date); D[s]=z.set_index('date').close
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# benchmark equal-weight over available names each day
bm=r.mean(axis=1)
# rolling beta, residual return, 20d reversal scaled by own 20d volatility
cov=r.rolling(60,min_periods=40).cov(bm)
var=bm.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0)
res=r.sub(beta.mul(bm,axis=0))
f=(-res.rolling(20,min_periods=15).sum()).div(r.rolling(20,min_periods=15).std())
# use signal at date t and forward return t+20; no lookahead
fr=p.shift(-20).div(p)-1
rows=[]
for d in f.index:
    a=f.loc[d]; y=fr.loc[d]; q=pd.concat([a,y],axis=1).dropna()
    if len(q)>=8:
        rows.append((d,q.iloc[:,0].corr(q.iloc[:,1]),len(q)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stats(x):
    return (len(x),x.n.mean(),x.ic.mean(),x.ic.std(ddof=1),x.ic.mean()/x.ic.std(ddof=1), (x.ic>0).mean())
print('assets',len(D),'date range',p.index.min(),p.index.max(),'IC dates',len(ic))
for label,sub in [('full',ic),('recent_3y',ic[ic.index>='2032-09-01']),('recent_2y',ic[ic.index>='2033-09-01']),('recent_1y',ic[ic.index>='2034-09-01']),('last180',ic.tail(180))]:
    print(label,stats(sub))
# alternative horizons same factor
for h in [10,20,40]:
    yy=p.shift(-h).div(p)-1; rr=[]
    for d in f.index:
        q=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
        if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1]))
    a=pd.Series(rr).dropna(); print('horizon',h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# artifact for deterministic audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_1_20350831_beta_residual_reversal_signal.csv',index=False)
ic.reset_index().to_csv('../persistent/miner_1_20350831_beta_residual_reversal_ic.csv',index=False)
