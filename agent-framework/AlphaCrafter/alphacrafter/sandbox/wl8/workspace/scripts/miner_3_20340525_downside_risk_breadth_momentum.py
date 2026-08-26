import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-05-24']
R=np.log(C).diff();
# medium-horizon trend divided by downside risk, with lagged breadth gate
mom=np.log(C/C.shift(60)).shift(1)
down=R.where(R<0).rolling(40,min_periods=20).std().shift(1).replace(0,np.nan)
raw=mom/down
breadth=(R.rolling(20,min_periods=10).mean()>0).mean(axis=1).shift(1)
# favor trend during broad participation, but retain sign and cross-sectional comparability
f=raw.mul((0.65+0.7*breadth).clip(.65,1.35),axis=0).rank(axis=1,pct=True)
f=f.sub(f.mean(axis=1),axis=0)
def q(h):return np.log(C.shift(-h)/C)
def calc(x):
 a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(ns,index=ds)
i,n=calc(q(10));print('end',C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4));print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4));print('turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20340525_downside_risk_breadth_momentum_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20340525_downside_risk_breadth_momentum_ic.csv')
