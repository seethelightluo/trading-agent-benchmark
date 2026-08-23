import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-02-21'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).dropna(); P=P.loc[:cutoff]; R=P.pct_change()
# downside deviation over trailing 30 observations, computed only from negative returns
neg=R.where(R<0); dd=neg.rolling(30,min_periods=3).std()*np.sqrt(30); mom=P/P.shift(20)-1; F=mom/dd

def evaluate(h):
 out=[]
 for i in range(45,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(out,columns=['date','ic','n']); print('horizon',h,'dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*15)); print('IC',x.ic.mean(),'ICIR',x.ic.mean()/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x.ic>0).mean()); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
for h in [5,10,20]: evaluate(h)
