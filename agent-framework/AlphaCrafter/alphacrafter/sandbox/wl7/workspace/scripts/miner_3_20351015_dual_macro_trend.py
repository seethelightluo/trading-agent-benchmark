import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2035-10-14']
ix='../persistent/index_data'
D=pd.read_csv(ix+'/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
V=pd.read_csv(ix+'/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(C.index).ffill()
r=C.pct_change(); vol=r.rolling(40,min_periods=25).std()
trend=(C/C.shift(20)-1).div(vol*np.sqrt(20))
# Dual macro regime: trend continuation in benign (weak DXY and low VIX), reversal in stressed (strong DXY or high VIX).
dweak=D<=D.rolling(60,min_periods=30).median()
vlow=V<=V.rolling(60,min_periods=30).median()
benign=(dweak & vlow); stressed=~benign
G=trend.mul(np.where(benign,1,-1),axis=0)
F=G.sub(G.median(axis=1),axis=0).shift(1)
F.to_csv('scripts/miner_3_20351015_dual_macro_trend_signal.csv',index_label='date')
def ev(h):
 y=C.shift(-h)/C-1; a=[]; ds=[]; ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(pd.Timestamp(d)); ns.append(len(q))
 a=np.asarray(a); ds=pd.DatetimeIndex(ds); ir=a.mean()/a.std(ddof=1)*np.sqrt(252)
 print(f'H{h} IC {a.mean():.8f} ICIR {ir:.8f} dates {len(a)} avgN {np.mean(ns):.2f} hit {np.mean(a>0):.4f}')
 for label,lo,hi in [('2027-30','2027','2030-12-31'),('2031-34','2031','2034-12-31'),('2035','2035','2035-10-14')]:
  z=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))]
  print(' ',label,len(z),f'IC {z.mean():.8f}' if len(z) else 'none')
for h in [1,5,10,20]: ev(h)
print('coverage',F.notna().sum().sum()/(15*len(F)),'turnover',np.nanmean(np.abs(F.diff()).mean(axis=1)))
print('dates',C.index.min(),C.index.max(),'assets',C.shape[1], 'benign',benign.mean())
