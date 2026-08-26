import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; macro='../persistent/index_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d['close'].astype(float)
close=pd.DataFrame(px).sort_index(); v=pd.read_csv(os.path.join(macro,'VIX.csv'),parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float).reindex(close.index).ffill()
vz=((v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std()).clip(-2,2).fillna(0)
r20=close/close.shift(20)-1
sig=-(r20).mul(1+0.50*vz,axis=0)
fwd=close.shift(-20)/close-1
rows=[]
for dt in sig.index[:-20]:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((dt,ic,len(z)))
a=np.array([x[1] for x in rows]); print('dates',len(a),'avg_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lo,hi in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-07-10')]:
 q=np.array([x[1] for x in rows if lo<=str(x[0].date())<=hi]); print(lo,hi,len(q),('IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0))) if len(q)>2 else 'NA')
print('turnover_proxy %.6f'%sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20310710_macro_stress_reversal_signal.csv',index=False)
