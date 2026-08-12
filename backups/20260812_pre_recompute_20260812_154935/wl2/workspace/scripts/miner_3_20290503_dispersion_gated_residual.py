import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').sort_index().close
P=pd.concat(D,axis=1).sort_index().ffill(); R=P.pct_change(); M=R.mean(axis=1)
# precompute 30d beta and residual returns, then lagged cumulative residual reversal
bet=pd.DataFrame(index=R.index,columns=R.columns,dtype=float)
for s in A:
 bet[s]=R[s].rolling(30).cov(M)/(M.rolling(30).var()+1e-8)
res=R-bet.mul(M,axis=0)
# dispersion of cross-asset 10d vol; gate high relative to trailing 120 observations
xvol=R.rolling(10).std().mean(axis=1); disp=xvol
med=disp.rolling(120,min_periods=20).median(); gate=disp>med
for hold in [3,5,10]:
 F=-res.rolling(hold).sum().div(R.rolling(30).std()+1e-6)
 vals=[]; ns=[]; turns=[]; prev=None
 for i in range(130,len(P)-1):
  if not gate.iloc[i]: continue
  z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); q=F.iloc[i].rank(pct=True); turns.append(np.abs(q-(prev if prev is not None else q)).mean()); prev=q
 x=np.asarray(vals)
 print('hold',hold,'dates',len(x),'active',int(gate.iloc[130:-1].sum()),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'dailyICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.mean(turns),4))
 for name,lo,hi in [('early',130,len(P)//2),('late',len(P)//2,len(P))]:
  a=[]
  for i in range(max(130,lo),min(len(P)-1,hi)):
   if gate.iloc[i]:
    z=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
    if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  print(name,'n',len(a),'ic',round(np.mean(a),6) if a else None)
