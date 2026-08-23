import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-05-16'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
P=pd.DataFrame({s:D[s]['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std()
# Range-efficiency reversal: recent return is penalized when directional persistence is high,
# while abnormal intraday close location identifies exhaustion. All windows are causal.
tr=pd.DataFrame({s:(D[s]['high']-D[s]['low']).reindex(P.index)/D[s]['close'].reindex(P.index) for s in U})
clv=pd.DataFrame({s:((D[s]['close']-D[s]['low'])-(D[s]['high']-D[s]['close'])).reindex(P.index)/(D[s]['high']-D[s]['low']).replace(0,np.nan).reindex(P.index) for s in U})
# exhaustion = negative 5d return plus adverse close-location, normalized by volatility
f=(-r.rolling(5).sum().div(v*np.sqrt(5)) - clv.rolling(3,min_periods=2).mean()*0.35)
# emphasize cross-sectional extremes only in high dispersion regimes
f=f.mul((r.rolling(5).std().mean(axis=1) / r.rolling(5).std().mean(axis=1).rolling(60,min_periods=30).median()).clip(.6,1.4),axis=0)
f.to_csv('scripts/miner_2_20290517_range_efficiency_exhaustion_signal.csv')
def run(h,a=None,b=None):
 xs=[]; cs=[]; ts=[]
 for i in range(len(P)-h):
  ds=str(P.index[i].date())
  if a and not(a<=ds<=b): continue
  z=pd.concat([f.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   xs.append(spearmanr(z.f,z.y).statistic);cs.append(len(z)/15)
   if i: ts.append(np.mean(np.sign(f.iloc[i].reindex(z.index))!=np.sign(f.iloc[i-1].reindex(z.index))))
 x=np.asarray(xs); return len(x),np.mean(cs),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(ts)
print('range',P.index.min(),P.index.max(),'assets',len(U),'rows',len(P))
for h in [1,5,10,15,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-05-16')]: print('regime',a,b,run(10,a,b))
print('artifact','scripts/miner_2_20290517_range_efficiency_exhaustion_signal.csv')
