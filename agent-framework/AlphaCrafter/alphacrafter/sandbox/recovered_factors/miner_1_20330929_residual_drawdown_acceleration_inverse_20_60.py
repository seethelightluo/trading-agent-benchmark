"""Miner 1 research: residual drawdown acceleration (20d vs 60d), tested as a simple contrarian recovery signal."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-09-28')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); resid=r-r.mean(axis=1)
# Each asset's loss from its own rolling residual-return wealth peak. A larger 20d than 60d drawdown means newly accelerated idiosyncratic damage; sign is contrarian recovery.
def dd(w):
 wealth=(1+resid).rolling(w,min_periods=w).apply(np.prod,raw=True)
 peak=wealth.rolling(w,min_periods=w).max()
 return wealth/peak-1
f=-(dd(20)-dd(60)).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE residual_drawdown_acceleration_inverse_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',f.dropna(how='all').shape[0],'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):out.append((d,z));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float); ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi];print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(ts)),6),'pairs',len(ts))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
f.to_pickle('scripts/miner_1_20330929_residual_drawdown_acceleration_inverse_20_60_signal.pkl')
print('INDEPENDENCE pending only if predictive gates pass; complete factor-library artifact mapping must be demonstrated before admission.')
