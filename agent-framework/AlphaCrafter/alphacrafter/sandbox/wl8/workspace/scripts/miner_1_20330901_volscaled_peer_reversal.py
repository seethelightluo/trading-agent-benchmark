import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-08-31']
r=np.log(C/C.shift(1)); vol=r.rolling(20,min_periods=15).std()
res=r.sub(r.mean(axis=1),axis=0)
f=(-res.rolling(5,min_periods=5).sum()/vol).rolling(3,min_periods=3).mean()
def run(h):
 fw=np.log(C.shift(-h)/C); vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(vals),pd.Series(ns)
i,n=run(10); j,m=run(10) # nonoverlap below
sel=f.index[::10]; jj=[]; mm=[]
fw=np.log(C.shift(-10)/C)
for d in sel:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:jj.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));mm.append(len(z))
j=pd.Series(jj);m=pd.Series(mm)
print('dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4),'start',i.index.min(),'end',i.index.max())
print('daily IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
print('nonoverlap dates',len(j),'avgN',round(m.mean(),3),'IC',round(j.mean(),6),'ICIR',round(j.mean()/j.std(ddof=1),6),'hit',round((j>0).mean(),4))
for w in [365,750,1260]:
 q=i.tail(w);print('recent',w,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10,20]:
 q=np.log(C.shift(-h)/C);a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(a),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330901_volscaled_peer_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20330901_volscaled_peer_reversal_ic.csv')
