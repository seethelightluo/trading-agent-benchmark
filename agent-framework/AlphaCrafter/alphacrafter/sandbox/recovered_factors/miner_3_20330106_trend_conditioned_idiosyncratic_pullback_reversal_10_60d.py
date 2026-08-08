"""One candidate: trend-conditioned idiosyncratic pullback reversal (10d/60d).
An asset earns a high score when it has a recent negative *market-residual* return,
but retains a positive 60-day total-return trend. This targets temporary pullbacks
inside established trends rather than unconditional short-term reversal. Inputs
are prior completed daily closes only; forward returns are evaluation labels.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-01-05')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
p=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:CUT]
r=p.pct_change(); m=r.mean(axis=1)
# 60d rolling market beta and beta-neutral residual returns
b=r.apply(lambda x:x.rolling(60,min_periods=42).cov(m)).div(m.rolling(60,min_periods=42).var()+1e-12,axis=0)
e=r-b.mul(m,axis=0)
res10=e.rolling(10,min_periods=8).sum()
resvol=e.rolling(20,min_periods=14).std()*np.sqrt(10)
trend=p.pct_change(60)
# continuous [0,1] trend gate: only persistent positive trends fully activate pullback signal
trend_gate=1/(1+np.exp(-8*trend.clip(-0.75,0.75)))
f=(-res10/(resvol+1e-12))*trend_gate
print('CANDIDATE trend_conditioned_idiosyncratic_pullback_reversal_10_60d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'coverage',round(float(f.notna().mean().mean()),6),'valid_cells',int(f.notna().sum().sum()))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 x=pd.Series(dict(vals),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 print('H%d IC=%.6f ICIR=%.6f dates=%d hit=%.4f meanN=%.2f'%(h,x.mean(),x.mean()/sd,len(x),(x>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=x.loc[lo:hi]; print('REGIME10',n,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
