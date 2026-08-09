import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:END]; V=pd.DataFrame({s:D[s].volume for s in U}).reindex(P.index)
r=P.pct_change(); dv=V.where(r<0,0).rolling(15,min_periods=10).sum(); uv=V.where(r>=0,0).rolling(15,min_periods=10).sum()
# lagged selling/buying volume imbalance, combined with lagged short reversal
asym=np.log((dv+1)/(uv+1)).clip(-3,3); f=-(P/P.shift(3)-1).shift(1)*(1+asym.shift(1).clip(lower=0))
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; obs=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: obs.append((pd.Timestamp(d),q.f.corr(q.y),len(q)))
 a=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avgN',round(a.n.mean(),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==1: print(a.assign(year=a.index.year).groupby('year').ic.agg(['mean','count']).round(6).to_string())
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',P.index.min(),P.index.max())
f.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_3_20261217_downside_volume_signal.csv',index=False)
