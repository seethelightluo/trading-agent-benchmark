import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-02-21')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change(); r10=p.pct_change(10)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Cross-sectional residual reversal, volatility normalized; lag one completed session.
csmed=r10.median(axis=1)
f=(-(r10.sub(csmed,axis='index')).div(vol.replace(0,np.nan))).shift(1)
for h in [5,10,20,40]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n'])
 ir=a.ic.mean()/(a.ic.std(ddof=1)+1e-12)
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('early',a[a.date<='2023-12-31']),('middle',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('late',a[a.date>='2027-01-01'])]:
  if len(sl): print(' ',name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
f.index.name='date'; f.to_csv('scripts/miner_1_20300221_vol_normalized_residual_reversal_signal.csv')
