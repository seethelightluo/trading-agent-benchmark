import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-10-04'); base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d[d.date<=cut].sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); ret=P.pct_change(); vol=ret.rolling(40,min_periods=25).std()*np.sqrt(40)
# medium trend, confirmed by short trend; risk normalize and cross-sectional demean
f=(0.65*(P/P.shift(60)-1)+0.35*(P/P.shift(20)-1))/vol
f=f.sub(f.mean(axis=1),axis=0)
F=P.shift(-10)/P-1
rows=[]; vals=[]
for dt in P.index:
 a=f.loc[dt]; b=F.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(a[ok],b[ok]).statistic,ok.sum())); vals.append(a)
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
print('dates',len(R),'range',R.index.min().date(),R.index.max().date(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),pd.DataFrame(vals,index=R.index).rank(pct=True).diff().abs().mean().mean()))
for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027-28',R.index>='2027-01-01'),('recent',R.index>=R.index.max()-pd.Timedelta(days=365))]:
 z=R.loc[m].ic; print(lab,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; z=[]
 for dt in P.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic)
 z=pd.Series(z); print('horizon',h,'IC %.6f ICIR %.6f n %d'%(z.mean(),z.mean()/z.std(),len(z)))
out=f.loc[R.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20281005_confirmed_trend_signal.csv',index=False)
