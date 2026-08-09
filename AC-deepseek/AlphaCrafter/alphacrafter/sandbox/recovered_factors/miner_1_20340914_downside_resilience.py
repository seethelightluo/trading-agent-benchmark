import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']); return d.set_index(d.date.dt.normalize()).close
P=pd.DataFrame({a:ld(a) for a in A}).sort_index(); R=np.log(P).diff(); m=R.mean(axis=1)
# One interpretable idea: downside resilience = negative conditional beta to market on recent down-market sessions
for w in [20,40,60]:
 for minneg in [8,12]:
  sig=pd.DataFrame(index=R.index,columns=A,dtype=float)
  down=m<0
  for i in range(len(R)):
   lo=max(0,i-w+1); ix=down.iloc[lo:i+1]; rr=R.iloc[lo:i+1]
   if ix.sum()>=minneg:
    mm=m.iloc[lo:i+1][ix]; den=(mm-mm.mean()).pow(2).sum()
    for a in A:
     yy=rr[a][ix]; sig.iloc[i,sig.columns.get_loc(a)]=((yy-yy.mean())*(mm-mm.mean())).sum()/den if den>1e-12 else np.nan
  # lower downside beta is better; lag one day
  x=-sig.shift(1); y=R.shift(-1); z=[]; ds=[]; ns=[]
  for d in P.index:
   ok=x.loc[d].notna()&y.loc[d].notna()
   if ok.sum()>=8 and x.loc[d,ok].nunique()>1:
    z.append(spearmanr(x.loc[d,ok],y.loc[d,ok]).statistic);ds.append(d);ns.append(ok.sum())
  z=np.array(z); ds=pd.DatetimeIndex(ds)
  print('W',w,'MIN',minneg,'dates',len(z),'N',round(np.mean(ns),2),'coverage',round(x.notna().mean().mean(),4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4),'turn',round((x.rank(axis=1)-x.rank(axis=1).shift(10)).abs().mean().mean()/14,4))
  for lo,hi in [('2020','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
   q=z[(ds>=lo+'-01-01')&(ds<=hi+'-12-31')]; print(' ',lo,round(q.mean(),4) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),4) if len(q)>1 else np.nan,len(q))
