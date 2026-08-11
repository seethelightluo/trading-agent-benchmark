import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for a in A}
r=pd.DataFrame({a:x.pct_change() for a,x in p.items()}).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].pct_change().reindex(r.index).ffill()
base=-(r.rolling(5,min_periods=4).sum()/r.rolling(20,min_periods=15).std())
# Penalize assets with high recent VIX beta; beta uses trailing 60 sessions, entirely lagged at signal date.
cov=r.rolling(60,min_periods=40).cov(v); vv=v.rolling(60,min_periods=40).var(); beta=cov.div(vv,axis=0)
# Cross-sectional standardized beta penalty, modest coefficient preserves reversal information.
beta_z=beta.sub(beta.mean(axis=1),axis=0).div(beta.std(axis=1),axis=0)
factor=base-0.20*beta_z
factor.to_csv('scripts/miner_1_20280210_vix_beta_reversal_signal.csv',index_label='date')
for h in [5,10,20]:
 vals=[]; ns=[]; ds=[]
 for i in range(100,len(r)-h):
  z=pd.concat([factor.iloc[i].rename('f'),r.iloc[i+1:i+1+h].sum(min_count=h-2).rename('fr')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.fr).statistic);ns.append(len(z));ds.append(r.index[i])
 x=np.array(vals); print(f'h={h} dates={len(x)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
 for s in ['2026-01-01','2027-01-01','2028-01-01']:
  q=x[np.array(ds,dtype='datetime64[ns]')>=np.datetime64(s)];print(s,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('turnover_proxy',np.nanmean([np.mean(np.abs(factor.iloc[i].rank(pct=True)-factor.iloc[i-10].rank(pct=True))) for i in range(110,len(factor),10)]))
