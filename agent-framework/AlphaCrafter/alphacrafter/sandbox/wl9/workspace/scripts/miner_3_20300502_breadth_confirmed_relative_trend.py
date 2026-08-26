import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-05-02')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change(); med=r.median(axis=1)
rel=(r.sub(med,axis=0)).rolling(40,min_periods=30).sum()
breadth=(r.rolling(10,min_periods=8).sum()>0).mean(axis=1)
confirm=2*breadth-1
vol=r.rolling(20,min_periods=15).std()
f=rel.div(vol+1e-8).mul(confirm,axis=0).shift(1)
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n'])
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 if h==40:
  for name,sl in [('2020-2023',a[a.date<='2023-12-31']),('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
   print(name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
f.index.name='date'; f.to_csv('scripts/miner_3_20300502_breadth_confirmed_relative_trend_signal.csv')
