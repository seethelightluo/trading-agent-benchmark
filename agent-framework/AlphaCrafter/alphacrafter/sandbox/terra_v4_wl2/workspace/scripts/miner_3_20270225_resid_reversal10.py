import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U},axis=1).sort_index();R=P.pct_change()
# Residualized medium reversal: remove common cross-asset return before accumulating 10d reversal.
common=R.median(axis=1); resid=R.sub(common,axis=0); f=(-resid.rolling(10).sum()).shift(1); y=P.shift(-1)/P-1
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).dropna();print('dates',len(x),'avgN',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'coverage',x.n.mean()/15)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-25')]:
 q=x.set_index('date').loc[lo:hi].ic;print('REG',lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=P.shift(-h)/P-1;a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print('H',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'assets',P.shape[1],'calendar_dates',P.shape[0])
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_3_20270225_resid_reversal10.csv',index=False)
