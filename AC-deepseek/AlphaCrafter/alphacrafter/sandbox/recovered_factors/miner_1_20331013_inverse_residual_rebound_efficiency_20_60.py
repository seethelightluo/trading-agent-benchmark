"""Miner 1: inverse residual rebound-efficiency factor (20d/60d). One factor idea."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-10-12')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
# Recovery efficiency is positive residual gains relative to preceding downside magnitude.
# Its inverse is an interpretable cross-asset contrarian signal: fragile, one-sided rebounds
# (large gains relative to earlier losses) are expected to mean revert.
up=resid.clip(lower=0).rolling(20,min_periods=14).sum()
down=(-resid.clip(upper=0)).rolling(60,min_periods=42).sum()
f=-(up/(down+1e-8)).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE inverse_residual_rebound_efficiency_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',f.dropna(how='all').shape[0],'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 out=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi];print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
f.to_pickle('scripts/miner_1_20331013_inverse_residual_rebound_efficiency_20_60_signal.pkl')
print('INDEPENDENCE pending only if predictive gates pass.')
