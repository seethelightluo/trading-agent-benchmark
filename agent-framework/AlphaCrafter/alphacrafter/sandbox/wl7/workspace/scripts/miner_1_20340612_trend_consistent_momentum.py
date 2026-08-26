import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x): x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Trend-consistent momentum: 20d return, rewarded only when 60d trend agrees;
# normalize by 40d volatility and lag one day.
m20=p/p.shift(20)-1; trend60=p/p.shift(60)-1; vol=r.rolling(40).std()*np.sqrt(40)
f=(m20/(vol+1e-12)*np.sign(trend60)).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;q=[];ns=[];ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(q,index=ds).dropna();print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
 if h==10:q10=q
for n in [180,500,750]:
 z=q10.iloc[-n:];print('RECENT%d H10 IC %.8f ICIR %.8f hit %.4f dates %d'%(n,z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z)))
print('period',p.index.min().date(),p.index.max().date(),'rows',len(p),'assets',len(p.columns));print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340612_trend_consistent_momentum_signal.csv',index=False)
print('artifact scripts/miner_1_20340612_trend_consistent_momentum_signal.csv')
