import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-03-29']
r=np.log(C).diff(); cs=r.mean(axis=1); idio=r.sub(cs,axis=0)
# lagged 8-session contrarian return, normalized by lagged idiosyncratic risk and smoothed
raw=-r.rolling(8,min_periods=6).sum().shift(1)
risk=idio.rolling(40,min_periods=20).std().shift(1)
disp=r.std(axis=1).rolling(10,min_periods=7).mean().shift(1); med=disp.rolling(252,min_periods=60).median()
gate=(disp/med).clip(.5,2.)
f=(raw/risk.replace(0,np.nan)).mul(gate,axis=0).ewm(span=5,min_periods=3,adjust=False).mean()
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
def forward(h): return np.log(C.shift(-h)/C)
def calc(x):
 vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return pd.Series(vals),pd.Series(ns)
i,n=calc(forward(10))
print('dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4))
print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,20]: print('decay',h,round(calc(forward(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340330_smoothed_high_disp_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340330_smoothed_high_disp_reversal_ic.csv')
