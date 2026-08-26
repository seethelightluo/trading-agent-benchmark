import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-09-19')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# Compression-reversal: medium return adjusted for realized risk, amplified when
# recent volatility is compressed relative to its medium-term baseline. Lagged.
ret20=p.pct_change(20)
vol20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
vol5=r.rolling(5,min_periods=4).std()*np.sqrt(252)
compression=(vol5/(vol20+1e-8)).clip(0,3)
raw=ret20/(vol20+1e-8)*compression
f=raw.rank(axis=1,pct=True).rolling(3,min_periods=2).mean().shift(1)
for h in [5,10,20,40,60]:
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((dt,q,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n'])
 if len(a):
  print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
  for name,sl in [('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2029',a[(a.date>='2027-01-01')&(a.date<='2029-12-31')]),('2030-YTD',a[a.date>='2030-01-01'])]:
   if len(sl): print('regime',name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_2_20300919_vol_compression_reversal_signal.csv')
