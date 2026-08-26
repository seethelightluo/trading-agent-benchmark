import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2033-09-14']
r=np.log(C).diff(); v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(-np.log(C/C.shift(5))/v).rolling(3,min_periods=3).mean(); q=np.log(C.shift(-10)/C)
def calc(mask):
 rows=[]
 for d in f.index[mask]:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 x=pd.Series([a[1] for a in rows],index=[a[0] for a in rows]); ns=pd.Series([a[2] for a in rows],index=x.index)
 return x,ns
all_i,n=calc(np.ones(len(f),dtype=bool)); dates=all_i.index
non=all_i.loc[[j%10==0 for j in range(len(all_i))]]
print('factor short_horizon_volnorm_reversal_10d');print('period',dates.min().date(),dates.max().date(),'dates',len(all_i),'nonoverlap',len(non),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
for label,x in [('daily',all_i),('nonoverlap',non)]: print(label,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=all_i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 qq=np.log(C.shift(-h)/C);a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],qq.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(a),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330915_short_horizon_volnorm_reversal_signal.csv',index=False)
all_i.rename('ic').to_csv('scripts/miner_1_20330915_short_horizon_volnorm_reversal_ic.csv')
