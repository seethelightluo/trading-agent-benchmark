import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:END]
for k in [1,2,5]:
 r=P.pct_change(k,fill_method=None).shift(1); resid=r.sub(r.median(axis=1),axis=0); vol=P.pct_change(fill_method=None).rolling(20,min_periods=15).std().shift(1); f=-resid/(vol+1e-12)
 for h in [1,5]:
  y=P.shift(-h).div(P)-1; obs=[]
  for d in P.index:
   q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8: obs.append(q.f.corr(q.y))
  a=np.array(obs); print('k',k,'h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 print('k',k,'coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
