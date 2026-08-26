import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float).replace(0,np.nan) for s in U}).sort_index().loc[:'2033-06-22']
lr=np.log(p).diff(); ret40=np.log(p/p.shift(40)); down=np.sqrt((lr.where(lr<0,0.0)**2).rolling(60,min_periods=30).mean())
raw=ret40/(down+1e-12); raw=raw.clip(raw.quantile(.10,axis=1),raw.quantile(.90,axis=1),axis=0); f=raw.rank(axis=1,pct=True).sub(.5).rolling(3,min_periods=3).mean()
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],np.log(p.shift(-h)/p).loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 x=pd.Series(vals); print('horizon',h,'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('coverage',np.mean([len(z)>=8 for z in []]) if False else f.notna().sum().sum()/(15*len(f)))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [365,750,1260]:
 vals=[]
 for d in f.index:
  z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=pd.Series(vals).tail(n);print('recent',n,'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330623_downside_risk_adjusted_momentum_signal.csv',index=False)
