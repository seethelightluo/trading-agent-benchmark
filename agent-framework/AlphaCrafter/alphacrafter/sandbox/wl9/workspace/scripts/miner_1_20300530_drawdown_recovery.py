import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-05-30'); D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# Recovery quality: rebound from 60d low, discounted when the rebound is noisy.
low=p.rolling(60).min(); rebound=p/low-1
noise=r.rolling(20).std()*np.sqrt(252)
sig=(rebound/(noise+0.01)).sub((rebound/(noise+0.01)).median(axis=1),axis=0).shift(1)
for h in [5,10,20,40]:
 y=p.shift(-h)/p-1; out=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append((dt,q,len(z)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 for nm,sl in [('middle',a.loc['2024-01-01':'2026-12-31']),('late',a.loc['2027-01-01':])]:print(' ',nm,len(sl),round(sl.ic.mean(),6),round(sl.ic.mean()/sl.ic.std(ddof=1),6))
print('turnover_proxy',round((sig.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
sig.index.name='date';sig.to_csv('scripts/miner_1_20300530_drawdown_recovery_signal.csv')
