import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-10-28']
ix='../persistent/index_data'
D=pd.read_csv(ix+'/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
V=pd.read_csv(ix+'/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
r=C.pct_change(); m=r.mean(axis=1)
# Residual short-horizon reversal: remove common cross-asset move, scale by recent risk.
res=r.sub(m,axis=0).rolling(5,min_periods=5).sum()
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
raw=(-res/vol).replace([np.inf,-np.inf],np.nan)
# Stable macro activation: reversal in stressed conditions, trend-following in benign conditions.
drank=D>D.rolling(60,min_periods=30).median(); vrank=V>V.rolling(60,min_periods=30).median()
stress=drank|vrank
F=raw.mul(np.where(stress,1.0,-0.35),axis=0)
F=F.sub(F.median(axis=1),axis=0).shift(1)
F.to_csv('scripts/miner_1_20351029_macro_residual_reversal_signal.csv',index_label='date')
def ev(h):
 y=C.shift(-h)/C-1; a=[]; ds=[]; ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(pd.Timestamp(d)); ns.append(len(q))
 a=np.asarray(a); ds=pd.DatetimeIndex(ds); ir=a.mean()/a.std(ddof=1)*np.sqrt(252)
 print(f'H{h} IC {a.mean():.8f} ICIR {ir:.8f} dates {len(a)} avgN {np.mean(ns):.2f} hit {np.mean(a>0):.4f}')
 for label,lo,hi in [('2027-30','2027','2030-12-31'),('2031-34','2031','2034-12-31'),('2035','2035-01-01','2035-10-28')]:
  z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]
  print(' ',label,len(z),f'IC {z.mean():.8f}' if len(z) else 'none')
for h in [1,5,10,20]: ev(h)
print('coverage',F.notna().sum().sum()/(15*len(F)),'turnover',np.nanmean(np.abs(F.diff()).mean(axis=1)))
print('dates',C.index.min(),C.index.max(),'assets',C.shape[1], 'stress',stress.mean())
