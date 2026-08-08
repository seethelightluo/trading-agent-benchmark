"""Miner 2 revalidation: one-observation volatility-scaled cross-asset reversal.
Uses only data visible at runtime; factor is lagged one completed daily session.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date')['close'],errors='coerce')
P=pd.DataFrame(C).sort_index(); R=P.pct_change()
# Higher: unusually negative last completed return relative to trailing own volatility.
f=(-(R/R.rolling(20,min_periods=15).std())).sub((-(R/R.rolling(20,min_periods=15).std())).median(axis=1),axis=0).shift(1)
cutoff=P.dropna(how='all').index.max(); fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,sl=None):
 x=f if sl is None else f.loc[sl[0]:sl[1]]; y=fw[h].reindex(x.index); vals=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   s=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(s): vals.append(s); nn.append(len(q))
 z=np.array(vals)
 return dict(ic_dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/z.std(ddof=1)),6),hit_ratio=round(float((z>0).mean()),6),mean_valid_names=round(float(np.mean(nn)),3),min_valid_names=int(min(nn))) if len(z) else {'ic_dates':0}
print('FACTOR volscaled_reversal_1obs CUTOFF',cutoff.date(),'INSTRUMENTS',len(A),'DATES',len(P))
print('SIGNAL_CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6))
for h in fw: print('HORIZON',h,stats(h))
for label,sl in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('2029_current',('2029-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME5',label,stats(5,sl))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'XSEC_STD',round(float(f.std(axis=1).mean()),6))
