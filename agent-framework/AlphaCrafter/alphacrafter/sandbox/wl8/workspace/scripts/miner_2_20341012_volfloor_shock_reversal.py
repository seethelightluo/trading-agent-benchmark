import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().groupby(level=0).last().loc[:'2034-10-11']
R=np.log(C).diff(); mom=R.rolling(5,min_periods=4).sum()
# A volatility floor prevents low-volatility assets from receiving unstable oversized reversal scores.
rv=R.rolling(20,min_periods=15).std(); den=rv.rolling(60,min_periods=30).mean().clip(lower=0.01)
raw=-(mom/den).shift(1); f=raw.rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[];n=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(n,index=ds)
i,nn=calc(q(10)); print('end',C.index.max().date(),'dates',len(i),'avgN',round(nn.mean(),3),'coverage',round(nn.mean()/15,4)); print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20341012_volfloor_shock_reversal_signal.csv',index=False); i.rename('ic').to_csv('scripts/miner_2_20341012_volfloor_shock_reversal_ic.csv')
