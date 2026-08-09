import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
v5=v.pct_change(5).reindex(P.index).ffill(); r5=P.pct_change(5); sh=v5.clip(-.5,.5)
f=(-r5.mul(1+sh,axis=0)).where(sh>0,-r5*.5)
Y=P.shift(-1).div(P)-1
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print(h,len(ic),a.n.mean(),ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print(yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_3_20261217_vix_conditioned_reversal_signal.csv', index_label='date')
print('ARTIFACT scripts/miner_3_20261217_vix_conditioned_reversal_signal.csv')
print('high',int((sh>0).sum()),'total',int(sh.notna().sum()))
# pooled correlations with plain reversal and residual reversal
plain=-r5
res=-(r5-r5.mean(axis=1))
for n,x in [('plain',plain),('residual',res)]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename(n)],axis=1).dropna(); print('corr',n,z.f.corr(z[n]))
