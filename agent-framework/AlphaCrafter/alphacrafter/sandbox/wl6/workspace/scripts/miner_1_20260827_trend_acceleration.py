import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; C={}
for s in U:
 d=pd.read_csv(f'{b}/{s}.csv');d.date=pd.to_datetime(d.date);C[s]=d.set_index('date').close.sort_index()
P=pd.DataFrame(C).sort_index();R=P.pct_change();F=R.rolling(10,min_periods=8).sum()-R.shift(10).rolling(20,min_periods=15).sum()
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); rows=[]
 for s in U: rows.append(pd.DataFrame({'date':P.index,'f':F[s].values,'y':Y[s].values}))
 A=pd.concat(rows,ignore_index=True).dropna(); out=[]
 for dt,g in A.groupby('date'):
  if len(g)>=8:out.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').ic
 print('h',h,'dates',len(q),'avg_names',round(A.groupby('date').size().loc[q.index].mean(),2),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),4),'coverage',round(len(A)/(len(P)*15),4))
 if h==1:
  for yr in range(2020,2027):
   z=q[q.index.year==yr]
   if len(z):print('regime',yr,len(z),round(z.mean(),5),round(z.mean()/z.std(ddof=1),5))
Rk=F.rank(axis=1,pct=True); print('turnover',round(Rk.diff().abs().mean(axis=1).dropna().mean(),4))
