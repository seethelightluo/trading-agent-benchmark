import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
px=pd.DataFrame({s:D[s].close for s in U}).sort_index()
# Medium-term trend, lagged one completed session and volatility-normalized.
r=px.pct_change(60); vol=px.pct_change().rolling(20).std(); sig=(r/vol).shift(1)
fwd=px.pct_change().shift(-1); out=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(out,columns=['date','n','ic']).set_index('date'); m=a.ic.mean(); ir=m/a.ic.std(ddof=1)*np.sqrt(252)
rank=sig.rank(axis=1,pct=True); t=[]
for x,y in zip(rank.index[:-1],rank.index[1:]):
 if x in a.index and y in a.index:
  q=pd.concat([rank.loc[x],rank.loc[y]],axis=1).dropna();t.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
print('dates',len(a),'avg_names',a.n.mean(),'coverage',a.n.mean()/15,'IC',m,'ICIR',ir,'hit',(a.ic>0).mean(),'turn',np.mean(t))
for yr in range(2020,2027):
 q=a[a.index.year==yr].ic
 if len(q)>1:print(yr,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
for w in [252,504,756]:
 q=a.tail(w).ic;print('recent',w,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
