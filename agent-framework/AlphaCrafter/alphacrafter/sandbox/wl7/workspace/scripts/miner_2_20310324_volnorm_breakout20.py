import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# volatility-compressed breakout: trend normalized by total path volatility, lagged
vol=r.rolling(20,min_periods=15).std(); f=(p.pct_change(20)/vol).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h); a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna(); print(h,len(a),round(a.mean(),8),round(a.mean()/a.std(),5),round((a>0).mean(),4))
a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
a=pd.Series(a); n=len(a); print('dates',len(p),'usable',n,'avgN',f.notna().sum(axis=1).mean(),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),5));print('regimes',[round(q.mean(),6) for q in [a.iloc[:n//3],a.iloc[n//3:2*n//3],a.iloc[2*n//3:]]])
