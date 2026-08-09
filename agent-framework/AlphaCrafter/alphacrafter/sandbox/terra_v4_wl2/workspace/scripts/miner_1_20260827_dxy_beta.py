import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-08-26')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for a in assets}; dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
R=pd.DataFrame({a:s.pct_change() for a,s in px.items()}); C=R.join(dxy.pct_change().rename('dxy'),how='inner').sort_index(); fwd=pd.DataFrame({a:px[a].pct_change().shift(-1) for a in assets}).reindex(C.index)
def calc(f,y):
 q=[]; ds=[]; ns=[]
 for d in C.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 return pd.Series(q,index=ds),ns
for w in [30,40,60,90,120]:
 b=C[assets].rolling(w,min_periods=max(25,w-15)).cov(C.dxy).div(C.dxy.rolling(w,min_periods=max(25,w-15)).var(),axis=0); f=-b
 ic,ns=calc(f,fwd); print('window',w,'dates',len(ic),'names',round(np.mean(ns),2),'coverage',round(f.stack().notna().mean(),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
 for h in [5,10]:
  y=pd.DataFrame({a:px[a].pct_change(h).shift(-h) for a in assets}).reindex(C.index); q,n=calc(f,y); print(' h',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',len(q))
 print('regime',ic.groupby(ic.index.year).mean().round(4).to_dict())
 for name,s in [('rev',-R.rolling(5).sum()),('mom',R.rolling(20).sum())]:
  z=pd.concat([f.stack().rename('f'),s.stack().rename('x')],axis=1).dropna();print(' corr',name,round(z.f.corr(z.x),4))
