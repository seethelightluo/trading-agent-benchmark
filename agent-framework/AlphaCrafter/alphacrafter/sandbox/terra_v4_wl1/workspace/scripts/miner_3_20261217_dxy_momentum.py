import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; ib='../persistent/index_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
dxy=pd.read_csv(f'{ib}/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(P.index).ffill()
# cross-sectional momentum, with a sign flip after strong dollar rallies (lagged, no lookahead)
r=P.shift(1)/P.shift(21)-1
z=(dxy.shift(1)/dxy.shift(21)-1).clip(-.08,.08)
# dollar-sensitive defensive tilt: penalize momentum during strong dollar regime
f=r*(1-1.5*z.values[:,None])
f=pd.DataFrame(f,index=P.index,columns=U)
f.to_csv('scripts/miner_3_20261217_dxy_momentum_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
