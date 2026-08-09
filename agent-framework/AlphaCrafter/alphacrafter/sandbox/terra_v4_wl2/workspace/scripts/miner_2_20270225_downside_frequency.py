import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame(px).pct_change()
# Downside shock frequency: favor assets with fewer negative return days, measured relative to the cross-section.
# Lagged trailing windows are known at the decision date.
fac=-(r.lt(0).astype(float).rolling(40,min_periods=30).mean())
fwd=pd.DataFrame(px).shift(-1)/pd.DataFrame(px)-1
rows=[]
for d in fac.index:
 z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(rows,columns=['date','n','ic'])
for h in [1]:
 print('H',h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15'),('2026-07-16','2027-02-24')]:
 z=q.set_index('date').loc[lo:hi].ic
 print('REGIME',lo,hi,'dates',len(z),'IC',z.mean() if len(z) else np.nan,'ICIR',z.mean()/z.std() if len(z)>1 else np.nan)
rank=fac.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean())
pd.DataFrame(fac.stack(),columns=['signal']).reset_index().rename(columns={'level_1':'asset'}).to_csv('../persistent/factor_signals_miner_2_20270225_downside_frequency.csv',index=False)
