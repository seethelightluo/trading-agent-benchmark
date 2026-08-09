import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
# Regime-conditioned medium-term trend: risk-adjusted 20d return, favor trend when VIX is falling, reverse mildly when rising.
vchg=v.pct_change(5).reindex(P.index).ffill().clip(-.5,.5)
raw=P.pct_change(20).shift(1).div(vol)
f=raw.mul(np.where(vchg.values[:,None]<0,1.0, -0.35))
f=f.sub(f.median(axis=1),axis=0)
Y=P.shift(-1).div(P)-1
rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
print('dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for yr,g in ic.groupby(ic.index.year): print('year',yr,'n',len(g),'ic',g.mean(),'icir',g.mean()/g.std(ddof=1))
for h in [5,10]:
 yy=P.shift(-h).div(P)-1; rr=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),yy.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rr.append(q.f.corr(q.y))
 z=pd.Series(rr).dropna(); print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20261217_regime_trend_signal.csv',index_label='date')
print('ARTIFACT scripts/miner_2_20261217_regime_trend_signal.csv')
for n,x in [('vixrev',-P.pct_change(5).mul(1+vchg,axis=0)),('plainmom',P.pct_change(20).shift(1).div(vol))]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename(n)],axis=1).dropna(); print('corr',n,z.f.corr(z[n]))
