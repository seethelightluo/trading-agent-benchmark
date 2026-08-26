import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Cross-sectional short-horizon reversal, scaled by idiosyncratic recent volatility and activated only when VIX is elevated vs its 60d median.
vol=r.rolling(20).std(); gate=(vix>vix.rolling(60).median()).astype(float); f=(-r.rolling(5).sum()/vol).mul(gate,axis=0).replace(0,np.nan); f=f.sub(f.mean(axis=1),axis=0).shift(1)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],p.pct_change(5).shift(-5).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z),len(z)/15))
o=pd.DataFrame(rows,columns=['date','ic','n','coverage']).set_index('date')
for w in [None,500,250]:
 q=o if w is None else o.tail(w); print(w,len(q),q.n.mean(),q.coverage.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
for a,b in [('2026','2029'),('2030','2033'),('2034','2035')]:
 q=o.loc[a:b]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
print('turnover',f.rank(pct=True).diff().abs().mean().mean()); o.to_csv('scripts/miner_1_20351001_vix_reversal_signal.csv')
