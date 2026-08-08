import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(p):
 d=pd.read_csv(p,parse_dates=['date']); return d.set_index(d.date.dt.normalize()).close
P=pd.DataFrame({a:ld('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index(); R=np.log(P).diff(); m=R.mean(axis=1)
v=ld('../persistent/index_data/VIX.csv').reindex(P.index).ffill(); vr=np.log(v).diff()
# One interpretable idea: resilience during VIX shocks, residualized to equal-weight market
for w in [20,40,60]:
 for th in [0.0,0.03,0.05]:
  shock=(vr>th).astype(float)
  # residual versus market contemporaneous return, estimated only trailing window
  mv=m.rolling(60,min_periods=30).var(); beta=pd.DataFrame(index=R.index,columns=A,dtype=float)
  for a in A: beta[a]=R[a].rolling(60,min_periods=30).cov(m)/mv
  resid=R-beta.multiply(m,axis=0)
  sig=(resid*shock).rolling(w,min_periods=max(10,w//2)).mean()
  f=R.shift(-1); z=[]; ds=[]; ns=[]
  for d in P.index:
   x,y=sig.loc[d],f.loc[d]; ok=x.notna()&y.notna()
   if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
    z.append(spearmanr(x[ok],y[ok]).statistic);ds.append(d);ns.append(ok.sum())
  z=np.array(z); ds=pd.DatetimeIndex(ds)
  print('W',w,'TH',th,'dates',len(z),'N',round(np.mean(ns),2),'cov',round(sig.notna().mean().mean(),3),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),3),'turn',round((sig.rank(axis=1,pct=True).diff(10).abs().mean().mean()),4))
  for lo,hi in [('2020','2025'),('2026','2029'),('2030','2032'),('2033','2034')]:
   q=z[(ds>=lo+'-01-01')&(ds<=hi+'-12-31')]; print(' ',lo,round(q.mean(),4) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),4) if len(q)>1 else np.nan,len(q))
