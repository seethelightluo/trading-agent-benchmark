import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(os.path.join(b,a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=np.log(p).diff()
# Short/intermediate trend with volatility compression: 15d momentum multiplied by recent-vs-long volatility ratio.
ret=r.rolling(15,min_periods=12).sum(); v10=r.rolling(10,min_periods=8).std(); v40=r.rolling(40,min_periods=30).std(); f=(ret*(v40/(v10+1e-9))).shift(1)
y=np.log(p).shift(-10)-np.log(p); vals=[]; ds=[]; ns=[]
for d in f.index:
 ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8:
  vals.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic); ds.append(d); ns.append(ok.sum())
ic=pd.Series(vals,index=ds).dropna(); print('dates',len(ic),'avgN',np.mean(ns),'coverage',f.loc[ds].notna().mean().mean()); print('IC %.8f ICIR %.8f hit %.4f turnover %.5f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for n in (60,120,252,756):
 z=ic.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for lo,hi in [('2026','2028'),('2029','2030'),('2031','2032')]:
 z=ic.loc[lo:hi]; print('regime',lo,hi,len(z),'%.6f %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
f.to_csv('scripts/miner_2_20320415_volcompression_trend_signal.csv')
