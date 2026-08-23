import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-06-13'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]
def macro(sym): return pd.read_csv('../persistent/index_data/'+sym+'.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
vix,dxy=macro('VIX'),macro('DXY'); r=P.pct_change()
rel20=r.rolling(20,min_periods=15).sum(); resid=rel20.sub(rel20.mean(axis=1),axis='index')
vs,ds=vix.pct_change(5),dxy.pct_change(5)
zv=(vs-vs.rolling(126,min_periods=60).mean())/(vs.rolling(126,min_periods=60).std()+1e-12); zd=(ds-ds.rolling(126,min_periods=60).mean())/(ds.rolling(126,min_periods=60).std()+1e-12)
stress=((zv.clip(-2,2)+zd.clip(-2,2))/2).clip(-2,2)
F=(resid.mul(1+0.35*stress.clip(lower=-1),axis='index')).rolling(3,min_periods=2).mean()
vol=r.rolling(20,min_periods=15).std(); F=F.div(vol+1e-8).clip(-5,5); F.to_csv('scripts/miner_3_20290614_stress_relative_momentum_signal.csv')
def run(h,a=None,b=None):
 vals=[];cov=[];turns=[]
 for i in range(len(P)-h):
  ds0=str(P.index[i].date())
  if a and not(a<=ds0<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);cov.append(len(z)/15)
   if i: turns.append(np.mean(np.sign(F.iloc[i].reindex(z.index))!=np.sign(F.iloc[i-1].reindex(z.index))))
 x=np.asarray(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
print('range',P.index.min(),P.index.max(),'assets',len(U),'dates',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-06-13')]: print('regime',a,b,run(5,a,b))
