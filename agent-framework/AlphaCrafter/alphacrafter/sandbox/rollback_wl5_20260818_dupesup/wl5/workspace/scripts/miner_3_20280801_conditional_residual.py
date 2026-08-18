import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];b=Path('../persistent/stock_data');P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:'2028-07-31'].ffill();R=P.pct_change();bm=R.mean(1); beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0); y=P.shift(-10)/P-1
for cond in ['all','up','down']:
 f=-(P.pct_change(3)-beta.mul(bm.rolling(3).sum(),axis=0)); vals=[]
 for dt in f.index:
  if cond=='up' and bm.loc[:dt].tail(5).sum()<=0:continue
  if cond=='down' and bm.loc[:dt].tail(5).sum()>0:continue
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals);print(cond,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
