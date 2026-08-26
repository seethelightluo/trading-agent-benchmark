import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().groupby(level=0).last().loc[:'2034-09-13']
R=np.log(C).diff(); res=R.sub(R.mean(axis=1),axis=0)
# Contrarian risk-adjusted intermediate residual trend: fade 40d residual move, scaled by downside volatility.
down=res.where(res<0,0.0)
raw=(-res.rolling(40,min_periods=25).sum()).shift(1)/down.rolling(40,min_periods=25).std().shift(1).replace(0,np.nan)
# robust rank signal, entirely lagged
f=raw.rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 x=np.log(C.shift(-h)/C); a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(ns,index=ds)
i,n=calc(10)
print('end',C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(h)[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340914_contrarian_downside_trend_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340914_contrarian_downside_trend_ic.csv')
