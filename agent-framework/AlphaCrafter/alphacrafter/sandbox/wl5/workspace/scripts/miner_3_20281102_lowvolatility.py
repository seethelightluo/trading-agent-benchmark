import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2028-11-01'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); px[s]=d[d.date<=cut].sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); f=-r.rolling(20,min_periods=15).std(); f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20]:
 F=P.shift(-h)/P-1; rows=[]
 for dt in P.index:
  a=f.loc[dt]; b=F.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   q=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(q): rows.append((dt,q,ok.sum()))
 R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
 print('horizon',h,'dates',len(R),'range',R.index.min().date(),R.index.max().date(),'avg_n %.2f coverage %.4f IC %.6f ICIR %.6f hit %.4f'%(R.n.mean(),R.n.mean()/15,x.mean(),x.mean()/x.std(),(x>0).mean()))
 for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent',R.index>=R.index.max()-pd.Timedelta(days=365))]:
  z=R.loc[m].ic; print(' ',lab,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
 if h==10:
  out=f.loc[R.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20281102_lowvolatility_signal.csv',index=False)
