import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float)
    except Exception as e: print('missing',s,e)
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff()
# Residual short-term reversal: remove contemporaneous common cross-asset movement,
# scale by idiosyncratic 20d volatility, and activate only in high-dispersion regimes.
common=r.mean(axis=1)
beta=r.rolling(60,min_periods=30).cov(common).div(common.rolling(60,min_periods=30).var(),axis=0)
res=r.sub(beta.mul(common,axis=0)); ivol=res.rolling(20,min_periods=12).std()*np.sqrt(20)
raw=-(np.log(p/p.shift(10))-beta.rolling(10,min_periods=5).sum().mul(common.rolling(10,min_periods=5).sum(),axis=0))/(ivol+1e-8)
disp=r.sub(r.mean(axis=1),axis=0).std(axis=1)
gate=disp.shift(1)>disp.shift(1).rolling(252,min_periods=126).quantile(.75)
f=raw.rolling(3,min_periods=2).mean().where(gate, np.nan)
rows=[]; signals=[]
for d in f.index:
 q=np.log(p.shift(-10)/p); z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
  for s in z.index: signals.append((d,s,f.loc[d,s]))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index(); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN %.3f coverage %.4f'%(ns.mean(),ns.mean()/15))
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for n in [365,750,1260]:
 z=i.tail(n); print('recent',n,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for start,end in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2032-12-31')]:
 z=i.loc[start:end]; print('regime',start[:4],len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
print('turnover %.6f'%raw.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC %.6f'%np.nanmean(a))
pd.DataFrame(signals,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20330203_beta_neutral_highdisp_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values,'n':ns}).to_csv('scripts/miner_1_20330203_beta_neutral_highdisp_reversal_ic.csv',index=False)
