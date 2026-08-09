import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]; R=P.pct_change(); r3=R.rolling(3,min_periods=3).sum(); breadth=R.gt(0).mean(axis=1); bshock=(breadth-.5).abs(); th=bshock.shift(1).rolling(120,min_periods=60).quantile(.7); gate=bshock>th
F=-r3.where(gate,0.0); F=F.sub(F.median(axis=1),axis=0)
for label,sl in [('full',slice(None)),('recent',slice(pd.Timestamp('2024-01-01'),None)),('post',slice(pd.Timestamp('2026-07-16'),None))]:
 f=F.loc[sl]; y=R.shift(-1).loc[f.index]; a=[]; ns=[]
 for d in f.index:
  x=f.loc[d]; z=y.loc[d]; ok=x.notna()&z.notna()&(x.nunique()>1)
  if ok.sum()>=8:
   q=spearmanr(x[ok],z[ok]).statistic
   if np.isfinite(q): a.append(q); ns.append(ok.sum())
 a=np.array(a); print(label,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(float(F.notna().mean().mean()),4),'turnover',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6)); F.to_csv('scripts/miner_3_20261217_breadth_extreme_reversal_signal.csv',index_label='date')
