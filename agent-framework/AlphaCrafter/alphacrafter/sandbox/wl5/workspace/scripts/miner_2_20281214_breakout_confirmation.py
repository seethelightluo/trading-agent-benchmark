import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2028-12-13'); px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); d=d[d.date<=cut]
 px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
hi=P.rolling(120,min_periods=80).max().shift(1); dist=P/hi-1
breadth=(r.rolling(20,min_periods=15).mean()>0).astype(float).rolling(20,min_periods=15).mean()-0.5
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
f=(dist*(1+1.5*breadth))/vol; f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20]:
 rows=[]
 fw=P.shift(-h)/P-1
 for dt in P.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): rows.append((dt,ic,len(z)))
 R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
 print('H',h,'dates',len(R),'avg_n',R.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean()))
 if h==10:
  for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent252',R.index>=R.index[-1]-pd.Timedelta(days=365))]:
   q=R.loc[m].ic; print(lab,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
rank=f.rank(axis=1,pct=True); print('range',P.index.min().date(),P.index.max().date(),'assets',len(px),'coverage',f.notna().sum(axis=1).mean()/15,'turnover',rank.diff().abs().mean().mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20281214_breakout_confirmation_signal.csv',index=False)
