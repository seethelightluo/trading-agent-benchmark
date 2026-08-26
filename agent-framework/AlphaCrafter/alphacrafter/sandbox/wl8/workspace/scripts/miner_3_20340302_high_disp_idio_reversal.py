import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-03-01']
r=np.log(C).diff()
# Short-horizon reversal, normalized by idiosyncratic (cross-sectional demeaned) risk,
# activated only when contemporaneous lagged dispersion is elevated relative to its own history.
raw=-r.rolling(5,min_periods=4).sum().shift(1)
csmean=r.mean(axis=1); idio=r.sub(csmean,axis=0)
risk=idio.rolling(30,min_periods=15).std().shift(1)
disp= r.std(axis=1).rolling(10,min_periods=7).mean().shift(1)
base=disp.rolling(252,min_periods=60).median()
gate=(disp/base).clip(0.5,2.0)
f=(raw/risk.replace(0,np.nan)).mul(gate,axis=0)
# winsorize cross-section for stability
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),pd.Series(ns)
i,n=calc(q(10));
print('factor high_disp_idio_reversal_5d dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w);print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,20]: print('decay',h,round(calc(q(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20340302_high_disp_idio_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_3_20340302_high_disp_idio_reversal_ic.csv')
