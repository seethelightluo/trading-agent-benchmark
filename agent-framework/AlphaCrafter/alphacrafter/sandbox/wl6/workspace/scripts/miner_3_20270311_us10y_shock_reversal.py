import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-10'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); m=R['US10Y']
z=(m-m.rolling(60,min_periods=40).mean())/m.rolling(60,min_periods=40).std()
vol=R.rolling(20,min_periods=15).std(); sig=(-R.shift(1)/vol.shift(1)).mul((z.shift(1).abs()>1.5).astype(float),axis=0)
rows=[]
for d in sig.index:
 f=sig.loc[d]; fw=R.shift(-1).loc[d]; ok=f.notna()&fw.notna()
 if ok.sum()>=8 and f[ok].nunique()>1: rows.append([d,ok.sum(),spearmanr(f[ok],fw[ok]).statistic])
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15)
for h in [1,5,10]:
 fw=R.rolling(h).sum().shift(-h+1); out=[]
 for d in sig.index:
  f=sig.loc[d]; y=fw.loc[d]; ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1: out.append(spearmanr(f[ok],y[ok]).statistic)
 q=pd.Series(out).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('regimes')
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-03-10')]:
 q=x.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('rank_turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean(),'active_dates',(sig.abs().sum(axis=1)>0).sum())
