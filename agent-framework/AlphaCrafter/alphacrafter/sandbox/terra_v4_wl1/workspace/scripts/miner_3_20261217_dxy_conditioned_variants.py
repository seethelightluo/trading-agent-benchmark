import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];E=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:E]
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill();tr=d.shift(1).pct_change(20);z=(tr-tr.rolling(252,min_periods=60).mean())/tr.rolling(252,min_periods=60).std()
r=P.pct_change(7).shift(1);v=P.pct_change().rolling(20,min_periods=10).std().shift(1);f=-(r.sub(r.median(axis=1),axis=0)).div(v)
for name,g in [('high',z>0),('low',z<=0),('extreme',z.abs()>1)]:
 x=f.mask(pd.DataFrame(np.broadcast_to(~g.values[:,None],f.shape),index=f.index,columns=f.columns));y=P.shift(-1).div(P)-1;a=[];ns=[]
 for dt in P.index:
  q=pd.concat([x.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.x.nunique()>1:a.append(spearmanr(q.x,q.y).statistic);ns.append(len(q))
 a=np.array(a);print(name,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(x.notna().sum().sum()/x.size,4))
