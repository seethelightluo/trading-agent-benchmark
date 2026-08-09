import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
# Signal at t uses only VIX through t-1 and asset prices through t-1.
r5=P.pct_change(5).shift(1); v5=v.pct_change(5).shift(1).reindex(P.index).ffill().clip(-.5,.5)
f=(-r5).mul((1+v5),axis=0).where(v5>0,(-r5).mul(.5,axis=0))
y=P.shift(-1).div(P)-1
rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((d,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']); ic=a.ic
print('dates',len(ic),'avgN',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for yr,g in a.groupby(a.date.dt.year): print('year',yr,'dates',len(g),'IC',g.ic.mean(),'ICIR',g.ic.mean()/g.ic.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20261217_vix5d_reversal_signal.csv',index=False)
print('period',P.index.min().date(),P.index.max().date(),'symbols',len(U))
