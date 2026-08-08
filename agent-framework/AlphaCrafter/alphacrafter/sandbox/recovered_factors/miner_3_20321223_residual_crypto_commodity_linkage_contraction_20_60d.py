"""One candidate: residual crypto-commodity linkage contraction (20d/60d).
Measures each asset's change in average beta-residual correlation to BTC, ETH,
WTI and COPPER; a linkage contraction is scored positively as a potential
cross-asset diversification/resilience signal. Completed-bar inputs only.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-12-22')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B=['BTC','ETH','WTI','COPPER']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
p=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:CUT]; r=p.pct_change(); m=r.mean(axis=1)
b=r.apply(lambda x:x.rolling(60,min_periods=42).cov(m)).div(m.rolling(60,min_periods=42).var()+1e-12,axis=0); e=r-b.mul(m,axis=0)
# each daily cross-sectional factor cell is negative current-vs-baseline linkage
c20=pd.DataFrame({a:pd.concat([e[a].rolling(20,min_periods=14).corr(e[z]) for z in B if z!=a],axis=1).mean(axis=1) for a in A})
f=-(c20-c20.rolling(60,min_periods=42).mean())
print('CANDIDATE residual_crypto_commodity_linkage_contraction_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'coverage',round(float(f.notna().mean().mean()),6),'valid_cells',int(f.notna().sum().sum()))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 print('H%d IC=%.6f ICIR=%.6f dates=%d hit=%.4f meanN=%.2f'%(h,x.mean(),x.mean()/sd,len(x),(x>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=x.loc[lo:hi]; print('REGIME10',n,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(ts)),6),'pairs',len(ts))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
