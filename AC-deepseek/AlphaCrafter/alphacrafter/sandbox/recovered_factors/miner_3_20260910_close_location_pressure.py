import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; F={}; R={}; V={}
for s in A:
 d=get_stock_daily_data(s,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 c=pd.to_numeric(d.close,errors='coerce'); h=pd.to_numeric(d.high,errors='coerce'); l=pd.to_numeric(d.low,errors='coerce')
 cl=(2*c-h-l)/(h-l).replace(0,np.nan)
 # persistent buying/selling pressure, smoothed and volatility scaled
 F[s]=cl.rolling(15,min_periods=10).mean()
 R[s]=c.pct_change(); V[s]=R[s].rolling(20,min_periods=15).std()
fac=pd.concat(F,axis=1); ret=pd.concat(R,axis=1); vol=pd.concat(V,axis=1)
# evaluate close-location pressure as raw signal and inverse (choose strongest)
print('FACTOR close_location_pressure_15 = mean((2C-H-L)/(H-L),15)')
print('history',fac.index.min(),fac.index.max(),'assets',len(A),'cells',fac.notna().sum().sum())
def ev(x,h):
 fw=ret.rolling(h).sum().shift(-h+1) # forward approx next h from t
 vals=[]; ns=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals); return len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),np.mean(ns)
for h in [1,5,10,20]: print('h',h,ev(fac,h))
# regimes 1d
for nm,mask in [('2020',fac.index.year==2020),('2021_22',(fac.index.year>=2021)&(fac.index.year<=2022)),('2023_24',(fac.index.year>=2023)&(fac.index.year<=2024)),('2025_26',fac.index.year>=2025)]:
 x=fac.loc[mask]; print('REGIME',nm,ev(x,1))
# turnover
rank=fac.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
# library correlations pooled
import glob,json
mx=0
for fn in glob.glob('factors/*.json'):
 if '.bak' in fn: continue
 d=json.load(open(fn)); fid=d['factor_id']; E={}
 for s in A:
  # reconstruct known library using definitions
  if fid=='miner_1_ravmom_20obs': E[s]=R[s].rolling(20).sum()
  elif fid=='miner_3_risk_adjusted_trend_20d': E[s]=R[s].rolling(20).sum()/V[s]
  elif fid=='miner_1_volnorm_reversal_5obs': E[s]=-R[s].rolling(5).sum()/R[s].rolling(10).std()
  elif fid=='miner_2_realized_volatility_20obs': E[s]=-V[s]
  elif fid=='miner_3_relative_volume_participation_20d': continue
  else: continue
 e=pd.concat(E,axis=1); z=pd.concat([fac.stack(),e.stack()],axis=1).dropna(); rho=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
 print('LIBCORR',fid,rho,'cells',len(z)); mx=max(mx,abs(rho))
print('MAX_ABS_LIBRARY_CORRELATION',mx)
