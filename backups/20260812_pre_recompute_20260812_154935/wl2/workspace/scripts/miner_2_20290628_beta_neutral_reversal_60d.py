import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').sort_index().close
P=pd.concat(D,axis=1).sort_index().ffill();R=P.pct_change();m=R.mean(axis=1)
# Variant: 60-session rolling beta neutralization, 3-session residual reversal, lagged one day.
rows=[]
for t in range(65,len(P)-1):
 v={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<40: continue
  b=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/np.var(z.iloc[:,1],ddof=1)
  vol=R[s].iloc[t-19:t+1].std()
  v[s]=-(R[s].iloc[t-2:t+1].sum()-b*m.iloc[t-2:t+1].sum())/vol if vol>1e-8 else np.nan
 f=pd.Series(v).dropna();z=pd.concat([f,R.iloc[t+1]],axis=1).dropna()
 if len(z)>=8:rows.append((P.index[t],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z),f))
x=np.array([r[1] for r in rows]);print('dates',len(x),'avgN',np.mean([r[2] for r in rows]),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'coverage',np.mean([r[2] for r in rows])/15)
for cut in ['2020-01-01','2023-01-01','2026-01-01','2028-01-01','2029-01-01']:
 y=x[[r[0]>=pd.Timestamp(cut) for r in rows]];print(cut,len(y),y.mean(),y.mean()/y.std(ddof=1))
turn=[];prev=None
for _,_,_,f in rows:
 q=f.rank(pct=True)
 if prev is not None:turn.append(q.sub(prev,fill_value=.5).abs().mean())
 prev=q
print('turnover',np.mean(turn))
for h in [3,5,10]:
 y=[]
 for date,_,_,f in rows:
  j=P.index.get_loc(date)
  if j+h<len(P):
   z=pd.concat([f,(P.iloc[j+h]/P.iloc[j]-1).reindex(f.index)],axis=1).dropna()
   if len(z)>=8:y.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 y=np.array(y);print('h',h,'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1))
