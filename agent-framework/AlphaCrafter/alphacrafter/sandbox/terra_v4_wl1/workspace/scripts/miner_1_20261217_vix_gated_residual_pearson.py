import pandas as pd,numpy as np
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}).loc[:END]
r=P.pct_change(); r7=P.pct_change(7); vol=r.rolling(20,min_periods=10).std(); med=r7.median(axis=1).where(r7.count(axis=1)>=8); base=-(r7.sub(med,axis=0)).div(vol)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:END].shift(1); high=(v>v.rolling(60,min_periods=30).median()).astype(float).reindex(P.index).fillna(0); f=base.mul(high,axis=0)
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; vals=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(q.f.corr(q.y))
 a=pd.Series(vals); print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
print('coverage',f.notna().sum().sum()/f.size)
f.stack().rename('signal').to_csv('scripts/miner_1_20261217_vix_gated_residual_signal.csv',header=True)
print('artifact scripts/miner_1_20261217_vix_gated_residual_signal.csv')
print('high dates',high.sum(),'total',high.notna().sum())
# pooled corr with existing simple reversal
print('corr base',f.stack().corr((-r7).stack()))
print('dates',len(P.index),'avgN',f.notna().sum(axis=1).mean())
# quarterly
