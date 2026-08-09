"""One candidate: low-dispersion downside beta-residual reversal (10d/20d).
Research cutoff is the last completed session before current date, 2032-09-15.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-09-15')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
p=pd.concat([load(a) for a in A],axis=1,sort=False).sort_index().loc[:CUT]
r=p.pct_change(); m=r.mean(axis=1)
beta=r.apply(lambda x:x.rolling(60,min_periods=42).cov(m)).div(m.rolling(60,min_periods=42).var()+1e-12,axis=0)
e=r-beta.mul(m,axis=0)
# Reversal is only expressed during broad declines with below-normal cross-asset dispersion:
# this separates orderly risk-off mean reversion from disorderly idiosyncratic shocks.
down=m.rolling(5,min_periods=5).sum()<0
disp=r.std(axis=1).rolling(5,min_periods=5).mean()
lowdisp=disp < disp.rolling(60,min_periods=42).median()
f=(-e.rolling(10,min_periods=8).sum().div(e.rolling(20,min_periods=14).std()+1e-12)).where(down & lowdisp)
print('CANDIDATE low_dispersion_downside_beta_residual_reversal_10_20d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('active_dates',int(f.notna().any(axis=1).sum()),'coverage',round(float(f.notna().mean().mean()),6),'valid_cells',int(f.notna().sum().sum()))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
 x=pd.Series(dict(vals),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print('H%d IC=%.6f ICIR=%.6f dates=%d hit=%.4f meanN=%.2f'%(h,x.mean(),x.mean()/sd,len(x),(x>0).mean(),np.mean(ns)))
 if h==1:
  for n,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01','2032-09-15')]:
   z=x.loc[lo:hi]
   print('REGIME1',n,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'hit',round((z>0).mean(),4) if len(z) else None)
ranks=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(ranks)):
 q=ranks.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: tos.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(tos)),6),'pairs',len(tos))
print('DECAY',{h:{'ic':round(float(x.mean()),6),'icir':round(float(x.mean()/x.std(ddof=1)),6),'dates':len(x)} for h,x in ics.items()})
