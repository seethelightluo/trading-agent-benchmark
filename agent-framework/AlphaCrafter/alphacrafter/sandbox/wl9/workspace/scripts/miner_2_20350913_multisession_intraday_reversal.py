import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
D={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
O=pd.concat({s:d['open'] for s,d in D.items()},axis=1); C=pd.concat({s:d['close'] for s,d in D.items()},axis=1)
O.columns=U; C.columns=U; C=C.sort_index(); O=O.reindex(C.index).loc[:'2035-09-12']; C=C.loc[:'2035-09-12']
intra=(C/O-1).replace([np.inf,-np.inf],np.nan)
f=-intra.rolling(5,min_periods=4).mean().shift(1)
f=f.clip(f.quantile(.02,axis=1),f.quantile(.98,axis=1),axis=0)
def calc(h):
 fr=C.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(ds)); return x,np.mean(ns)
print('universe',len(U),'period',C.index.min().date(),C.index.max().date())
for h in [1,5,10,20,40]:
 x,n=calc(h); print(f'H{h} dates {len(x)} avgN {n:.2f} IC {x.mean():.6f} ICIR {x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit {(x>0).mean():.4f}')
x,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print(f'REG {a}-{b} dates {len(q)} IC {q.mean():.6f} ICIR {q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
print('coverage',f.notna().sum(axis=1).div(15).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20350913_multisession_intraday_reversal_signal.csv',index=False)
