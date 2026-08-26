import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-06-13')
prices={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 prices[s]=d.loc[d.index<=cut,'close'].astype(float)
p=pd.DataFrame(prices).sort_index(); logr=np.log(p).diff()
r3=logr.rolling(3,min_periods=3).sum(); vol=logr.rolling(20,min_periods=15).std(); trend=logr.rolling(60,min_periods=40).sum(); raw=-r3/vol
def sigrow(x):
 z=pd.concat([x,trend.loc[x.name]],axis=1).dropna()
 if len(z)<8 or z.iloc[:,1].std()==0:return pd.Series(np.nan,index=x.index)
 a=z.iloc[:,0];b=z.iloc[:,1]; beta=np.cov(a,b,ddof=1)[0,1]/b.var(ddof=1)
 return (a-beta*b-a.median()).reindex(x.index)
sig=raw.apply(sigrow,axis=1)
out=[]
for h in [1,5,10,20]:
 f=np.log(p.shift(-h)/p); vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:vals.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
 q=pd.DataFrame(vals,columns=['date','n','ic']);out.append((h,q))
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
r=sig.rank(axis=1,pct=True);print('coverage',round(sig.notna().sum().sum()/(sig.shape[0]*len(U)),5),'rank_turnover',round(r.diff().abs().mean(axis=1).dropna().mean(),5),'instruments',len(U),'dates',len(sig))
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20320614_smooth_residual_reversal_signal.csv',index=False)
