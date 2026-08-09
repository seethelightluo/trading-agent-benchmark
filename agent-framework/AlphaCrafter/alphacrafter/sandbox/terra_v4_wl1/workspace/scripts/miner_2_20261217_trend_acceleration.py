import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
# Decision t uses through t-1. Acceleration contrasts recent 20d return to prior 40d trend, scaled by lagged volatility.
r=P.pct_change(); r20=P.shift(1)/P.shift(21)-1; r60=P.shift(1)/P.shift(61)-1
vol=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(252)
f=(r20-r60/3).div(vol.replace(0,np.nan))
# cross-sectional demean removes common trend exposure
f=f.sub(f.mean(axis=1),axis=0)
f.to_csv('scripts/miner_2_20261217_trend_acceleration_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',P.index.min(),P.index.max())
# pooled overlap against common momentum and reversal artifacts if present
for fn in ['scripts/miner_2_20260716_risk_adjusted_momentum_20d_signal.csv','scripts/miner_1_20261217_ewm_reversal_signal.csv']:
 try:
  x=pd.read_csv(fn,index_col='date',parse_dates=True); z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('corr',fn,z.f.corr().iloc[0,1])
 except Exception as e: print('corr unavailable',fn,e)
print('ARTIFACT scripts/miner_2_20261217_trend_acceleration_signal.csv')
