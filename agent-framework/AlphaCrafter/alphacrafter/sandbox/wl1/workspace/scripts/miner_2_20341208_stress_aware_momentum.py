import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def read(s,kind='stock_data'):
 return pd.read_csv('../persistent/'+kind+'/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
D={s:read(s) for s in U}; p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change(); v=read('VIX','index_data').reindex(p.index).ffill()
mom=r.rolling(20).sum(); vchg=v.pct_change(10).clip(-1,1); mult=(1-2*vchg).clip(-1.5,2.5)
f=mom.mul(mult,axis=0).shift(1)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],r.iloc[i+1:i+11].sum()],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15)
print('overall IC %.6f ICIR %.6f hit %.3f turnover %.5f'%(df.ic.mean(),df.ic.mean()/df.ic.std(ddof=1),(df.ic>0).mean(),f.rank(pct=True).diff().abs().mean().mean()))
for a,b in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-12-31')]:
 q=df.loc[a:b].ic; print(a,'n',len(q),'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for h in [5,10,20,40]:
 rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8:rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'icir',np.nanmean(rr)/np.nanstd(rr,ddof=1),'n',len(rr))
f.to_csv('scripts/miner_2_20341208_stress_aware_momentum_signal.csv',index_label='date')
