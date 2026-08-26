import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-08-08')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
down=(-r.clip(upper=0)).rolling(40,min_periods=20).sum(); absall=r.abs().rolling(40,min_periods=20).sum()
f=(down/(absall+1e-12)).shift(1)
for h in [10,20,40,60]:
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((dt,q,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n'])
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 if h==40:
  for name,sl in [('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2029',a[(a.date>='2027-01-01')&(a.date<='2029-12-31')]),('2030-YTD',a[a.date>='2030-01-01'])]:
   if len(sl): print('regime',name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
  print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_3_20300808_downside_pressure_signal.csv')
