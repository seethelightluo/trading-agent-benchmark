import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-03-01']
r=np.log(C).diff(); m=r.mean(axis=1)
# Medium-term market-neutral momentum: each asset's 40d return minus its rolling
# 60d beta to the equal-weight benchmark times benchmark return. Inputs are lagged.
rm=r.rolling(40,min_periods=30).sum().shift(1)
beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0).shift(1)
market=m.rolling(40,min_periods=30).sum().shift(1)
res=rm-beta.mul(market,axis=0)
# Volatility normalization and 5-session smoothing, all using information through t-1.
vol=r.rolling(40,min_periods=30).std().shift(1)
f=res.div(vol.replace(0,np.nan)).rolling(5,min_periods=5).mean()
def calc(h):
 y=np.log(C.shift(-h)/C); vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return pd.Series(vals),pd.Series(ns)
i,n=calc(10)
print('factor beta_neutral_volnorm_momentum_40d_10d dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6), 'hit',round((x>0).mean(),4))
for h in [1,5,20]: print('decay',h,round(calc(h)[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20340302_beta_neutral_volnorm_momentum_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_2_20340302_beta_neutral_volnorm_momentum_ic.csv')
