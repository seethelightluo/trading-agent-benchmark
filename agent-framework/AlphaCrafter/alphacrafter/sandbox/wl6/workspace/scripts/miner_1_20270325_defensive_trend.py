import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-24'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); r20=R.rolling(20,min_periods=15).sum(); v=R.rolling(20,min_periods=15).std()
# lagged relative trend, with defensive benchmark trend as a smooth regime anchor
D=R[['XAU','US10Y','CN10Y']].rolling(20,min_periods=15).sum().mean(axis=1)
cs=r20.sub(r20.median(axis=1),axis=0)
sig=(cs/v).shift(1).mul((D>0).shift(1).astype(float),axis=0)
for gate in ['on','off']:
 S=sig if gate=='on' else (cs/v).shift(1); rows=[]
 for d in S.index:
  f=S.loc[d]; y=R.shift(-1).loc[d]; ok=f.notna()&y.notna()
  if ok.sum()>=8 and f[ok].nunique()>1: rows.append([d,ok.sum(),spearmanr(f[ok],y[ok]).statistic])
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); print('\nGATE',gate,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4))
 for h in [1,5,10]:
  fw=R.rolling(h).sum().shift(-h+1); q=[]
  for d in S.index:
   f=S.loc[d]; y=fw.loc[d]; ok=f.notna()&y.notna()
   if ok.sum()>=8 and f[ok].nunique()>1:q.append(spearmanr(f[ok],y[ok]).statistic)
  q=pd.Series(q).dropna(); print('h',h,'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),4))
 print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
 for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-03-24')]:
  q=x.loc[lo:hi].ic; print(label,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5) if len(q)>1 else 'nan')
