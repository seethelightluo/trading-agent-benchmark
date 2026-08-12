import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; end=pd.Timestamp('2031-11-27')
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
D=pd.concat(px,axis=1).sort_index()
# path-efficiency trend: signed 20d move / total absolute daily moves, with volatility scaling
ret=D.pct_change()
lag=D.shift(1)
r=lag.pct_change()
move=lag/lag.shift(20)-1
path=r.abs().rolling(20,min_periods=15).sum()
fac=move/path
# damp extreme noise using 60d vol; interpretable efficiency only
# fac itself is the candidate
fwd=D.shift(-1)/D-1
rows=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# date windows / regimes
print('candidate=path_efficiency_20d; dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.5f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-27','2026-01-01','2027-12-31'),('2028-30','2028-01-01','2030-12-31'),('2031','2031-01-01','2031-11-27'),('recent120',None,None)]:
 q=r.tail(120) if name=='recent120' else r.loc[a:b]
 print(name,len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
out='scripts/miner_2_20311127_path_efficiency_signal.csv'
fac.loc[r.index].to_csv(out)
print('signal',out)
