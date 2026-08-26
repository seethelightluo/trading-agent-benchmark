import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-01-18']
R=np.log(C).diff()
# Observation-only VIX; all macro inputs are lagged before use.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(C.index).ffill()
vp=v.rolling(252,min_periods=126).rank(pct=True).shift(1)
# Risk-adjusted medium momentum, cross-asset residualized, stress gated.
market=R.mean(axis=1)
res=R.sub(market,axis=0)
trend=res.rolling(40,min_periods=30).sum().shift(1)
down=R.where(R<0).rolling(40,min_periods=20).std().shift(1).replace(0,np.nan)
f=trend.div(down).mul((1+0.6*(vp-0.5)).clip(0.7,1.3),axis=0)
f=f.rolling(3,min_periods=3).mean()
def calc(h):
 x=np.log(C.shift(-h)/C);ics=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(ics),pd.Series(ns)
i,n=calc(10)
print('factor macro_stress_residual_trend_40d dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(h)[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340119_macro_stress_residual_trend_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340119_macro_stress_residual_trend_ic.csv')
