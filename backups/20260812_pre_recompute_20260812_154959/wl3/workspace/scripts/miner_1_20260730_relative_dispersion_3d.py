import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'; px={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):px[a]=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
P=pd.DataFrame(px); R=P.pct_change(); rel=R.rolling(3,min_periods=3).sum(); F=-rel.sub(rel.median(axis=1,skipna=True),axis=0)
# forward next observed row per asset (calendar next row); approximate via shifted -1
for h in [1,5,10]:
 fw=sum(R.shift(-k) for k in range(1,h+1)); vals=[]; ns=[]; ds=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z));ds.append(d)
 s=pd.Series(vals,index=ds).dropna(); print(h,len(s),np.mean(ns),np.mean(ns)/15,s.mean(),s.mean()/s.std(ddof=1),np.mean(s>0))
print('turn',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=pd.DataFrame(F.stack(),columns=['signal']);out.index.names=['date','symbol'];out.to_csv('scripts/miner_1_20260730_relative_dispersion_3d_signal.csv')
