import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-10-18'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].set_index('date').close.sort_index()
# forward-fill only historical observations to align asynchronous asset calendars; no future values are introduced
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); ret10=P/P.shift(10)-1
down=r.where(r<0).rolling(20,min_periods=12).std();disp=r.rolling(20,min_periods=12).std().mean(axis=1)
gate=(disp/disp.rolling(120,min_periods=60).median()).clip(0.5,2.0);f=-ret10.div(down*np.sqrt(10)).mul(gate,axis=0)
rows=[]; sig=[]
for dt in P.index:
 b=P.shift(-10).loc[dt]/P.loc[dt]-1;a=f.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:
  q=spearmanr(a[ok],b[ok]).statistic
  if np.isfinite(q): rows.append((dt,q,ok.sum()));sig.append(a)
R=pd.DataFrame(rows,columns=['date','ic','n']);R['date']=pd.to_datetime(R.date);R=R.set_index('date');x=R.ic
print('dates',len(R),'range',R.index.min().date(),R.index.max().date(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),pd.DataFrame(sig,index=R.index).rank(pct=True).diff().abs().mean().mean()))
for lab,m in [('2020-24',(R.index>='2020-01-01')&(R.index<'2025-01-01')),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent',R.index>=pd.Timestamp('2027-10-18'))]:
 z=R.loc[m].ic;print(lab,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
for h in [5,20]:
 fr=P.shift(-h)/P-1;z=[]
 for dt in P.index:
  a=f.loc[dt];b=fr.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:
   q=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(q):z.append(q)
 z=pd.Series(z);print('horizon',h,'IC %.6f ICIR %.6f n %d'%(z.mean(),z.mean()/z.std(),len(z)))
out=f.loc[R.index].stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20281019_dispersion_gated_reversal_signal.csv',index=False)
