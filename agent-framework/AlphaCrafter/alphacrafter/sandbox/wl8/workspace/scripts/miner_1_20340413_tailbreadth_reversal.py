import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-04-12'];r=np.log(C).diff();res=r.sub(r.mean(axis=1),axis=0)
shock=res.rolling(5,min_periods=4).sum().shift(1); vol=res.rolling(40,min_periods=25).std().shift(1); q=res.rolling(60,min_periods=30).quantile(.1).shift(1)
tail=res.lt(q); breadth=tail.rolling(10,min_periods=7).mean().mean(axis=1).shift(1)
# Activate reversal when recent cross-asset downside tail breadth is elevated; smooth gate.
gate=((breadth-breadth.rolling(252,min_periods=60).median())*8+0.5).clip(0.1,1.5)
f=(-shock/vol.replace(0,np.nan)).mul(gate,axis=0); f=f.rank(axis=1,pct=True);f=(f+f.shift(1)+f.shift(2))/3;f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 x=np.log(C.shift(-h)/C);a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
i,n=calc(10);print('period',C.index.min().date(),C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4));print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(h)[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340413_tailbreadth_reversal_signal.csv',index=False);i.rename('ic').to_csv('scripts/miner_1_20340413_tailbreadth_reversal_ic.csv')
