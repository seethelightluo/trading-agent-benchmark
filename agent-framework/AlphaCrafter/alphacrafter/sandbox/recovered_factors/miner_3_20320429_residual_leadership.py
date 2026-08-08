import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.set_index('date').close
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); b=R['SPX']; bm=b.rolling(60).mean(); bv=((b-bm)**2).rolling(60).mean()
beta=R.apply(lambda x: ((x.rolling(60).mean()-x.rolling(60).mean()*0) * 0))
# equivalent rolling covariance / variance
beta=pd.DataFrame({a:((R[a]*b).rolling(60).mean()-R[a].rolling(60).mean()*bm)/bv for a in A})
res=R.rolling(20).sum()-beta*b.rolling(20).sum().values[:,None]
rows=[]
for i in range(80,len(P)-1):
 z=pd.concat([res.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('dates',len(d),'meanN',d.n.mean(),'coverage',d.n.sum()/(len(d)*15))
for h in [1,5,10,20]:
 q=[]
 for i in range(80,len(P)-h):
  z=pd.concat([res.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('h',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
print('turnover10',np.nanmean((res.rank(axis=1,pct=True)-res.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1)))
for s,e in [('2024','2028'),('2028','2031'),('2031','2033')]:
 q=d.loc[s:e].ic;print(s,e,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('recent120',d.ic.tail(120).mean(),d.ic.tail(120).mean()/d.ic.tail(120).std(ddof=1))
