import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for a in A}
r=pd.DataFrame({a:x.pct_change() for a,x in p.items()}).sort_index()
# Trend persistence: medium return times fraction of positive sessions, scaled by realized volatility.
ret20=r.rolling(20,min_periods=18).sum(); persistence=r.gt(0).rolling(20,min_periods=18).mean(); vol20=r.rolling(20,min_periods=18).std()
factor=ret20*persistence/vol20
factor=factor.replace([np.inf,-np.inf],np.nan)
# Explicit lag: factor at date t predicts returns after t.
for h in [5,10,20]:
 vals=[]; ns=[]; ds=[]
 for i in range(100,len(r)-h):
  z=pd.concat([factor.iloc[i].rename('f'),r.iloc[i+1:i+1+h].sum(min_count=h-2).rename('fr')],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.f,z.fr).statistic); ns.append(len(z)); ds.append(r.index[i])
 x=np.asarray(vals); print(f'h={h} dates={len(x)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
 for s in ['2024-01-01','2026-01-01','2027-01-01','2028-01-01']:
  q=x[np.asarray(ds,dtype='datetime64[ns]')>=np.datetime64(s)]
  print(' ',s,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
turn=[]
for i in range(110,len(factor),10):
 a=factor.iloc[i].rank(pct=True); b=factor.iloc[i-10].rank(pct=True); turn.append(np.nanmean(abs(a-b)))
print('turnover_proxy',np.nanmean(turn),'valid_factor_coverage',factor.notna().mean().mean())
factor.to_csv('scripts/miner_1_20280224_trend_persistence_signal.csv',index_label='date')
