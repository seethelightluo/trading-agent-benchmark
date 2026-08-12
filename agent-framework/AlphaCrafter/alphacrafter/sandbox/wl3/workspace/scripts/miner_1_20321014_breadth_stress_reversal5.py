import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),usecols=['date','close']); d=d.drop_duplicates('date').set_index('date')['close'].astype(float); px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); mom20=p.pct_change(20); breadth=mom20.gt(0).mean(axis=1)
rev5=-p.pct_change(5); vol20=r.rolling(20).std()
f=rev5.mul((1.5-breadth).clip(0.25,1.5),axis=0).div(vol20).rolling(5).mean().shift(1)
fwd=p.shift(-10)/p-1; rows=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'assets',len(U),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean(),f.diff().abs().mean().mean()))
for n in [120,252,756]:
 x=q.tail(n); print('recent',n,'IC %.6f ICIR %.6f hit %.4f dates %d'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(),len(x)))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a,b,'n',len(x),'IC %.5f ICIR %.5f'%(x.ic.mean(),x.ic.mean()/x.ic.std()))
print('decay')
for h in [1,3,5,10,20]:
 yy=p.shift(-h)/p-1; z=[]
 for dt in p.index:
  t=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(t)>=8:z.append(spearmanr(t.iloc[:,0],t.iloc[:,1]).statistic)
 print(h,round(float(np.nanmean(z)),6),round(float(np.nanmean(z)/np.nanstd(z)),6))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('factors/miner_1_20321014_breadth_stress_reversal5_signal.csv',index=False)
