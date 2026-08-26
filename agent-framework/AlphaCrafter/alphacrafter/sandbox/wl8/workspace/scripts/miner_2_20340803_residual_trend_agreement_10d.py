import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-08-02'].ffill()
R=np.log(C).diff(); m=R.mean(axis=1); E=R.sub(m,axis=0)
# Lagged medium-term residual trend, risk-normalized and stabilized by recent trend agreement.
trend=E.rolling(30,min_periods=20).sum().shift(1)
vol=E.rolling(30,min_periods=20).std().shift(1)
short=E.rolling(10,min_periods=7).sum().shift(1)
f=(trend/vol.replace(0,np.nan)) * (1+0.35*np.sign(trend)*np.sign(short))
f=f.rank(axis=1,pct=True).sub(0.5,axis=0)
def forward(h): return np.log(C.shift(-h)/C)
def calc(x):
 vals=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 return pd.Series(vals,index=ds),pd.Series(ns,index=ds)
i,n=calc(forward(10)); print('end',C.index.max().date(),'dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4));print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]:print('decay',h,round(calc(forward(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20340803_residual_trend_agreement_10d_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_2_20340803_residual_trend_agreement_10d_ic.csv')
