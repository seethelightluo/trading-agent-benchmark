import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-03-13'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); bm=R.mean(axis=1)
beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0)
res=P.pct_change(10)-beta.mul(bm.rolling(10).sum(),axis=0); f=-res.sub(res.median(axis=1),axis=0); y=P.shift(-10)/P-1
def calc(lo=None,hi=None):
 a=[];ns=[]
 for d in f.loc[lo:hi].index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
print('10d beta residual reversal',calc())
for l,h in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-03-13')]:print(l,calc(l,h))
r=f.rank(axis=1,pct=True);print('turn',np.nanmean((r-r.shift()).abs().mean(axis=1)),'coverage',f.notna().mean().mean());f.to_csv('scripts/miner_3_20280314_beta_residual10_signal.csv')
