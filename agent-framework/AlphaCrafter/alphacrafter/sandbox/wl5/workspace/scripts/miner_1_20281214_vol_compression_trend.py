import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date)
 px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); P=P.loc[P.index<=pd.Timestamp('2028-12-13')]
r=P.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Trend rewarded when current volatility is compressed relative to its medium-term baseline.
f=P.pct_change(20)*(1+np.clip(1-v20/(v60+1e-8),-0.5,0.5))
f=f.sub(f.median(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in P.index:
 a=f.loc[dt];
 for_h=[]
 for h in [5,10,20]:
  b=P.iloc[P.index.get_loc(dt)+h]/P.loc[dt]-1 if P.index.get_loc(dt)+h<len(P) else pd.Series(index=U,dtype=float)
  ok=a.notna()&b.notna()
  if ok.sum()>=8: for_h.append((h,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 if for_h: rows.append((dt,for_h))
for h in [5,10,20]:
 q=[x for dt,ls in rows for x in ls if x[0]==h]; z=pd.DataFrame(q,columns=['h','ic','n']);
 print('horizon',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
 for lab,m in [('2020-24',P.index[:len(z)]<pd.Timestamp('2025-01-01')),('2025-26',(z.index>=0)&(z.index<0)),('2027-28',P.index[-1]>=pd.Timestamp('2027-01-01')),('recent',True)]: pass
# recompute clean horizon 10 and save artifact
h=10; out=[]
for i,dt in enumerate(P.index[:-h]):
 a=f.loc[dt];b=P.iloc[i+h]/P.iloc[i]-1;ok=a.notna()&b.notna()
 if ok.sum()>=8:
  out.append((dt,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
R=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');
for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent',R.index>=R.index.max()-pd.Timedelta(days=365))]:
 z=R.loc[m].ic; print(lab,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
f.loc[R.index].stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281214_vol_compression_signal.csv',index=False)
