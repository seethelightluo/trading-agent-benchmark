import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x
close=pd.DataFrame({s:D[s]['close'] for s in U}); ret=close.pct_change()
# one interpretable candidate: medium momentum weighted by directional consistency, lagged
r20=close/close.shift(20)-1
cons=(ret.gt(0).rolling(20).mean()-0.5)*2
vol=ret.rolling(20).std()
sig=(r20*cons/vol).shift(1)
# signal artifact
sig.reset_index().to_csv('scripts/miner_1_20270423_agreement_momentum_signal.csv',index=False)
rows=[]
for h in [1,5,10,20]:
 fwd=close.shift(-h)/close-1
 vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=np.array([v[1] for v in vals]);
 rows.append((h,len(a),np.mean([v[2] for v in vals]),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
print('candidate=agreement_momentum; dates range',sig.index.min(),sig.index.max())
for x in rows: print('h=%d dates=%d avg_n=%.2f IC=%+.8f ICIR=%+.8f hit=%.4f'%(x))
# regime split at 2023/25 for h5
h=5; fwd=close.shift(-h)/close-1; vals=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-12-31')]:
 q=np.array([v[1] for v in vals if str(v[0])[:4]>=a and str(v[0])[:10]<=b]); print('regime',a,b,'n',len(q),'IC',np.mean(q) if len(q) else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',sig.notna().mean().mean(),'turnover',((sig.rank(axis=1,pct=True).diff().abs()).mean().mean()))
