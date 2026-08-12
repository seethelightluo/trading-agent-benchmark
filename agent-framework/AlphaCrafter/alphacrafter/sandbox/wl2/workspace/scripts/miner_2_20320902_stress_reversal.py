import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
C={}
for s in U:
 p=os.path.join(base,s+'.csv')
 x=pd.read_csv(p)
 date=pd.to_datetime(x['date'])
 C[s]=pd.Series(x['close'].astype(float).values,index=date)
C=pd.DataFrame(C).sort_index(); R=C.pct_change()
# Lagged stress-conditioned short-term reversal: yesterday-known 5d return reversal,
# activated only when prior cross-asset breadth is weak; scale by trailing 20d volatility.
r5=R.rolling(5).sum(); vol=R.rolling(20).std()*np.sqrt(20)
breadth=(r5>0).mean(axis=1)
stress=(breadth.shift(1)<0.40)
f=(-r5/vol.replace(0,np.nan)).where(stress.shift(1).fillna(False))
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-10).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:
  rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(C.columns),'price_dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a,b,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [1,3,5,10]:
 rr=R.rolling(h).sum().shift(-h); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC %.6f n %d'%(np.nanmean(z),len(z)))
q=o.tail(120); print('recent120 IC %.6f ICIR %.6f n %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
print('turnover_proxy %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_2_20320902_stress_reversal_signal.csv',index_label='date')
