import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2029-10-03']
r=P.pct_change(); sig=(P.pct_change(20)/(r.rolling(20,min_periods=15).std()*np.sqrt(20))).shift(1)
def evalh(h):
 fwd=P.shift(-h)/P-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); return q
q=evalh(10)
print('period',q.index.min().date(),q.index.max().date(),'dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit', (q.ic>0).mean(),'turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,sub in [('2026',q.loc['2026']),('2027-28',q.loc['2027':'2028']),('recent360',q.tail(360)),('recent180',q.tail(180))]: print(name,'dates',len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1))
for h in [1,5,10,20]:
 a=evalh(h).ic.to_numpy(); print('decay',h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'obs',len(a))
out=sig.stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_1_20291004_volnorm_momentum_20d_signal.csv',index=False)
