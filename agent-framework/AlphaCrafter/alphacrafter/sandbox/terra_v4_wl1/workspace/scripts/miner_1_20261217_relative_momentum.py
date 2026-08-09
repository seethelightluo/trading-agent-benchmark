import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float) for s in U}).sort_index()
P=P[P.index<=cut]; r=P.pct_change(); y=P.shift(-1).div(P)-1
# Relative-strength persistence: 20d return relative to contemporaneous cross-sectional median, lagged one completed bar.
r20=P.pct_change(20); csmed=r20.median(axis=1); f=r20.sub(csmed,axis=0).shift(1)
obs=[]
for dt in P.index:
 q=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
 if len(q)>=8: obs.append((dt,q.f.corr(q.y),len(q)))
a=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date'); z=a.ic
print('cut',cut.date(),'dates',len(z),'avgN',a.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for yr,g in z.groupby(z.index.year): print('year',yr,'dates',len(g),'IC',g.mean(),'ICIR',g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'rankturn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_1_20261217_relative_momentum_signal.csv',index=False)
