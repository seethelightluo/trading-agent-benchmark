import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float)
    except Exception as e: print('missing',s)
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff()
mom=np.log(p/p.shift(40)); vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
# Contrarian form: assets with unusually strong recent risk-adjusted trends are faded.
f=(-mom/(vol+1e-8)).rolling(3,min_periods=2).mean()
rows=[]; signals=[]
for d in f.index:
    q=np.log(p.shift(-10)/p)
    z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
    if len(z)>=8:
        c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if pd.notna(c): rows.append((d,c,len(z)))
        for s in z.index: signals.append((d,s,f.loc[d,s]))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index(); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN %.3f'%ns.mean(),'coverage %.4f'%(ns.mean()/15))
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for start,end in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2032-12-31')]:
 z=i.loc[start:end]; print('regime',start[:4],len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1), (z>0).mean()))
print('turnover %.6f'%f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC %.6f'%np.nanmean(a))
pd.DataFrame(signals,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20321209_inv_volnorm_mom40_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values,'n':ns}).to_csv('scripts/miner_3_20321209_inv_volnorm_mom40_ic.csv',index=False)
