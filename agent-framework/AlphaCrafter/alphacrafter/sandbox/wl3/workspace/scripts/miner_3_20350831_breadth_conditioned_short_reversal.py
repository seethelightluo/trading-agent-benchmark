import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change(); vol=R.rolling(30,min_periods=20).std()
bread=R.rolling(5).median().shift(1); raw=-R.rolling(3).sum()/(vol*np.sqrt(3)+1e-12); F=raw.where(bread<0,0.0)
rows=[]
for h in [1,5,10,20]:
  ics=[]; turns=[]; cov=[]
  for i in range(40,len(P)-h):
   x=F.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:
    ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
    if i>40: turns.append((z.iloc[:,0].rank()!=F.iloc[i-1].reindex(z.index).rank()).mean())
  a=pd.Series(ics).dropna(); rows.append((h,len(a),a.mean(),a.mean()/a.std(),(a>0).mean(),np.mean(cov),np.mean(turns)))
print('dates',len(P),'instruments',len(D)); print('h,n,IC,ICIR,hit,cov,turn')
for x in rows: print(x)
F.index.name='date'; F.to_csv('scripts/miner_3_20350831_breadth_conditioned_short_reversal_signal.csv')
