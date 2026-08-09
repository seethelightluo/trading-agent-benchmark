import pandas as pd, numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date').close
vl=v.shift(1); state=(vl>vl.rolling(60,min_periods=30).median()).astype(float)
F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 r=d.close.pct_change(); rng=(d.high-d.low)/d.close; scale=rng.rolling(20,min_periods=10).median()
 # Lagged VIX regime: fade shock in high VIX; follow shock in calm VIX, normalized by typical range.
 F[a]=(-r*(2*state.reindex(d.index)-1)/(scale+1e-8)).clip(-20,20)
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_vix_range_switch_signal.csv')
print('assets',len(assets),'rows',len(fac))
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(fac.index); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=pd.DatetimeIndex(ds)); print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
