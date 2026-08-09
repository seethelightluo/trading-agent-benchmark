import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index();R=P.pct_change();m=R.mean(axis=1)
b=R.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0);e=R.sub(b.mul(m,axis=0),axis=0)
for w in [3,5,15,30]:
 F=-e.rolling(w,min_periods=max(3,w//2)).sum().shift(1);F=F.sub(F.median(axis=1),axis=0);a=[]
 for i in range(len(P)-1):
  z=pd.concat([F.iloc[i],P.iloc[i+1]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a);print('w',w,'n',len(a),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(ddof=1),7),'hit',round((a>0).mean(),4),'turn',round(F.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),5))
