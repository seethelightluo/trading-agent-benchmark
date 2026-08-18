import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-07-31'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); bm=R.mean(1); beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0)
y=P.shift(-10)/P-1
for h in [1,2,4,5,6,8,10]:
 f=-(P.pct_change(h)-beta.mul(bm.rolling(h).sum(),axis=0)); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
# smooth blend ranks 1d+3d+5d
fs=[]
for h in [1,3,5]:fs.append((-(P.pct_change(h)-beta.mul(bm.rolling(h).sum(),axis=0))).rank(axis=1,pct=True))
f=sum(fs)/3; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=np.array(vals);print('blend',len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)); f.to_csv('scripts/miner_3_20280801_resblend_signal.csv')
