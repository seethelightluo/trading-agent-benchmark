import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
R=pd.DataFrame({a:P[a].pct_change() for a in A})
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].pct_change().reindex(R.index)
var=dxy.rolling(60,min_periods=40).var()
beta=pd.DataFrame({a:R[a].rolling(60,min_periods=40).cov(dxy)/var for a in A})
raw=R.rolling(20,min_periods=20).sum(); F=raw-beta.shift(1).mul(dxy.rolling(20,min_periods=20).sum(),axis=0)
rows=[];sig=[]
for d in R.index:
 vals=F.loc[d];good=vals.dropna()
 if len(good)<8:continue
 z=vals-good.median()
 for a in A:sig.append((d,a,z.get(a,np.nan)))
 for h in [1,5,10]:
  ff=[];yy=[]
  for a in good.index:
   ix=P[a].index.get_loc(d)
   if ix+h<len(P[a]):ff.append(z[a]);yy.append(P[a].iloc[ix+h]/P[a].iloc[ix]-1)
  if len(ff)>=8:rows.append((d,h,spearmanr(ff,yy).statistic,len(ff)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h];print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=x.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal');print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6));pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_3_20270225_dxy_resid_momentum.csv',index=False)
