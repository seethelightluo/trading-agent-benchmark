import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index(); R=np.log(C).diff(); market=R.mean(axis=1); res=R.sub(market,axis=0)
# Yield stress is lagged percentile of absolute 20-session moves in the two tradable yield series.
yld=R[['US10Y','CN10Y']].mean(axis=1)
stress=yld.abs().rolling(20,min_periods=10).sum().rolling(252,min_periods=126).rank(pct=True).shift(1)
f=-res.rolling(10,min_periods=8).sum().shift(1).mul((0.7+0.9*stress).clip(0.7,1.6),axis=0).rolling(3,min_periods=3).mean()
ics=[]
for d in f.index:
 y=np.log(C.shift(-10)/C); z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
i=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date'); q=i.ic
print('dates',len(i),'avgN',round(i.n.mean(),2),'coverage',round(f.notna().mean().mean(),4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,z.mean(),z.mean()/z.std(ddof=1))
for h in [1,5,10,20]:
 y=np.log(C.shift(-h)/C); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
out=f.loc[i.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20350315_yield_stress_residual_reversal_signal.csv',index=False)
i.reset_index().to_csv('scripts/miner_1_20350315_yield_stress_residual_reversal_ic.csv',index=False)
