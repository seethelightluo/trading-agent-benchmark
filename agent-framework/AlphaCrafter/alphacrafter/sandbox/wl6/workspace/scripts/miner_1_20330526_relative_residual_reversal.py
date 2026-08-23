import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.pct_change() for a in A}
R=pd.DataFrame(R).sort_index(); m=R.mean(axis=1)
f=R.rolling(20,min_periods=15).sum().sub(m.rolling(20,min_periods=15).sum(),axis=0)*-1
F=R.shift(-1).rolling(10,min_periods=10).sum().shift(-9); x=[]; n=[];turn=[]; old=None
for d in f.index:
 z=pd.concat([f.loc[d],F.loc[d]],axis=1).dropna()
 if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
 r=f.loc[d].rank(pct=True)
 if old is not None:turn.append((r-old).abs().mean())
 old=r
s=pd.Series(x).dropna();print('dates',len(s),'avg_n',np.mean(n),'coverage',np.mean(n)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turnover',np.mean(turn))
for h in [5,10,20,40]:
 q=[];fh=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1))
 for d in f.index:
  z=pd.concat([f.loc[d],fh.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('decay',h,q.mean(),q.mean()/q.std())
