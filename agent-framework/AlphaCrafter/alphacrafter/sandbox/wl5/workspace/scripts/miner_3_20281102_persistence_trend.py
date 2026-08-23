import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date)
 px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Trend persistence: medium return weighted by directional consistency, normalized by recent risk.
cons=r.rolling(40,min_periods=30).mean()/r.rolling(40,min_periods=30).std()
f=(P.pct_change(60)*cons)/r.rolling(40,min_periods=30).std()
f=f.sub(f.mean(axis=1),axis=0)
rows=[]; sig=[]
for h in [5,10,20]:
 F=P.shift(-h)/P-1; rows=[]
 for dt in P.index:
  a=f.loc[dt]; b=F.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
 print('horizon',h,'dates',len(R),'range',R.index.min().date(),R.index.max().date(),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean()))
 for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent',R.index>=R.index.max()-pd.Timedelta(days=365))]:
  z=R.loc[m].ic
  print(' ',lab,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
 if h==10:
  sig=f.loc[R.index].stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_3_20281102_persistence_trend_signal.csv',index=False)
