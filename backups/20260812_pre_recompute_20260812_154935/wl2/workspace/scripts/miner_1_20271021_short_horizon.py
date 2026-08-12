import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index().loc[:'2027-10-20']
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); fwd=p.shift(-1)/p-1
cands={'rev1':-r,'rev2':-r.rolling(2,min_periods=2).sum(),'rev3':-r.rolling(3,min_periods=3).sum(),'rev5':-r.rolling(5,min_periods=5).sum(),'mom10':r.rolling(10,min_periods=8).sum(),'riskadj_rev3':-r.rolling(3,min_periods=3).sum()/r.rolling(20,min_periods=15).std()}
for name,raw in cands.items():
 f=raw.shift(1).ewm(span=3,min_periods=3,adjust=False).mean(); rows=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rows); print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
