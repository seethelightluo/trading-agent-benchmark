import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-04-12')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r60=p.pct_change(60)
vol=r.rolling(60).std()*np.sqrt(252)+1e-12
breadth=r.gt(0).rolling(20).mean()
# Medium-horizon momentum, risk scaled and rewarded for persistent positive paths; lag one day.
sig=(r60/vol)*(0.5+0.5*breadth); sig=sig.shift(1)
def calc(h):
 f=p.shift(-h)/p-1; rows=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
q=calc(10)
print('dates',len(q),'avg_names',q.n.mean(),'coverage',q.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10,20,40]:
 x=calc(h);print('decay',h,'IC %.8f ICIR %.8f dates %d'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),len(x)))
for a,b in [('2025','2026'),('2027','2029'),('2030','2034')]:
 x=q.loc[a:b];print('regime',a,b,len(x),'IC %.8f ICIR %.8f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()))
sig.tail(1).T.to_csv('scripts/miner_1_20340413_recovery_momentum_60d_signal.csv')
