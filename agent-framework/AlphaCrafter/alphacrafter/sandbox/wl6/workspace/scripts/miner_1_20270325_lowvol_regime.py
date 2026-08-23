import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-24');P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); V=R.rolling(20,min_periods=15).std(); market=R.mean(axis=1).rolling(20,min_periods=15).mean(); S=(-V).shift(1).mul((market<0).shift(1).astype(float),axis=0)
for name,sig in [('bear',S),('all',(-V).shift(1))]:
 rows=[]
 for d in sig.index:
  f=sig.loc[d]; y=R.shift(-1).loc[d];ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1:rows.append([d,ok.sum(),spearmanr(f[ok],y[ok]).statistic])
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');print('\n',name,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4))
 for h in [1,5,10]:
  fw=R.rolling(h).sum().shift(-h+1);q=[]
  for d in sig.index:
   f=sig.loc[d];y=fw.loc[d];ok=f.notna()&y.notna()
   if ok.sum()>=8 and f[ok].nunique()>1:q.append(spearmanr(f[ok],y[ok]).statistic)
  q=pd.Series(q);print('h',h,'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),4))
 print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
