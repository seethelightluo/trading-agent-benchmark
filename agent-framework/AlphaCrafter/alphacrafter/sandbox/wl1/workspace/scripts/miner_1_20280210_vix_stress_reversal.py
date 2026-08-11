import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for a in assets}
r=pd.DataFrame({a:s.pct_change() for a,s in px.items()}).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].reindex(r.index).ffill()
vol=r.rolling(20,min_periods=15).std(); raw=r.rolling(5,min_periods=4).sum(); base=-(raw/vol.replace(0,np.nan))
state=(vix > vix.rolling(60,min_periods=40).median()).astype(float)
factor=base.mul(0.75+0.75*state,axis=0)
factor.to_csv('scripts/miner_1_20280210_vix_stress_reversal_signal.csv',index_label='date')
for h in [5,10,20]:
 vals=[]; ns=[]; ds=[]
 for i in range(80,len(r)-h):
  z=pd.concat([factor.iloc[i].rename('f'),r.iloc[i+1:i+1+h].sum(min_count=h-2).rename('fr')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.fr).statistic); ns.append(len(z)); ds.append(r.index[i])
 x=np.asarray(vals); dsa=np.asarray(ds,dtype='datetime64[ns]'); ir=x.mean()/x.std(ddof=1)
 print(f'h={h} dates={len(x)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={x.mean():.6f} ICIR={ir:.6f} hit={np.mean(x>0):.4f}')
 for s in ['2025-01-01','2026-01-01','2027-01-01','2028-01-01']:
  q=x[dsa>=np.datetime64(s)]; print(f' {s} n={len(q)} IC={q.mean() if len(q) else np.nan:.6f} ICIR={q.mean()/q.std(ddof=1) if len(q)>1 else np.nan:.6f}')
ranks=factor.rank(axis=1,pct=True); changed=[]
for i in range(10,len(ranks),10):
 z=pd.concat([ranks.iloc[i-10],ranks.iloc[i]],axis=1).dropna(); changed.append(np.mean(np.abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover_proxy_10d=',np.mean(changed),'valid_signal_dates=',factor.dropna(how='all').shape[0])
