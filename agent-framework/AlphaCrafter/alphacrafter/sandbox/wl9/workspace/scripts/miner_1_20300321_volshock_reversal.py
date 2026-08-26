import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-03-21')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change(); r5=p.pct_change(5)
v5=r.rolling(5,min_periods=4).std(); v20=r.rolling(20,min_periods=15).std()
shock=(v5/(v20+1e-12)).clip(0,4)
# Reversal after recent move, emphasizing assets undergoing an unusual volatility shock.
f=(-r5*shock).shift(1)
rows=[]
for h in [5,10,20,40]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n'])
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4),'turnover_proxy',round(f.diff().abs().mean().mean(),6))
 if h==10:
  for name,sl in [('2020-2023',a[a.date<='2023-12-31']),('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
   print(name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6),'hit',round((sl.ic>0).mean(),4))
out=f; out.index.name='date'; out.to_csv('scripts/miner_1_20300321_volshock_reversal_signal.csv')
