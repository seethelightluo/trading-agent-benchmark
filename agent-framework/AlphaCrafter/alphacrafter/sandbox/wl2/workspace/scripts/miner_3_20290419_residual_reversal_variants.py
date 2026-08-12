import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').sort_index().close
P=pd.concat(D,axis=1).sort_index().ffill();R=P.pct_change();M=R.mean(axis=1)
def signal(i,win=30,hold=3):
 out={}
 for s in A:
  r=R[s].iloc[:i+1].tail(win);m=M.iloc[:i+1].tail(win)
  beta=r.cov(m)/(m.var()+1e-8);res=r-beta*m
  out[s]=-res.tail(hold).sum()/(r.std()+1e-6)
 return pd.Series(out)
for hold in [3,5,10]:
 vals=[]; ns=[]; turns=[]; prev=None
 for i in range(35,len(P)-1):
  f=signal(i,30,hold);y=R.iloc[i+1];z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));q=f.rank(pct=True);turns.append(np.abs(q-(prev if prev is not None else q)).mean());prev=q
 x=np.array(vals);print('hold',hold,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'dailyICIR',round(x.mean()/x.std(ddof=1),6),'annualICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(x>0),4),'turn',round(np.mean(turns),4))
