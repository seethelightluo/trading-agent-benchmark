"""Miner_3 single-idea test: high-dispersion broad-drawdown residual pullback reversal.
A short residual pullback is ranked only after simultaneous (ex-ante) high cross-asset
dispersion and a negative five-day equal-weight market move; otherwise no signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-05-25')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); ew=r.mean(axis=1); resid=r.sub(ew,axis=0)
pull=resid.rolling(5,min_periods=4).sum()
# State uses observations ending t: 20d mean daily CS dispersion in top 30% of trailing 60d,
# coupled with an equal-weight 5d completed-bar drawdown.
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); hi=disp.rolling(60,min_periods=45).rank(pct=True)>=.70
mkt5=(1+ew).rolling(5,min_periods=4).apply(np.prod,raw=True)-1
state=hi & (mkt5<0)
f=(-pull).where(state, np.nan).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE high_dispersion_broad_drawdown_residual_pullback_reversal_5_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('state_dates',int(state.sum()),'state_fraction',round(float(state.mean()),6),'valid_dates',int(f.notna().any(axis=1).sum()),'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat((f.loc[d].rename('f'),fw.loc[d].rename('y')),axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float); ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi_ in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi_]; print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turns)),6),'pairs',len(turns))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
f.to_pickle('scripts/miner_3_20330526_high_dispersion_broad_drawdown_residual_pullback_reversal_5_20_60_signal.pkl')
