import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].reindex(P.index).ffill()
# lagged macro regime: VIX level relative to trailing 252d median, fully observable at decision
vz=v.shift(1)/v.shift(1).rolling(252,min_periods=60).median()-1
# reversal is amplified in elevated volatility, damped in calm volatility
r3=P.pct_change(3).shift(1)
g=(1+0.75*vz.clip(-.5,.5)).clip(.25,1.75)
f=-r3.mul(g,axis=0)
print('dates',len(P),'instruments',len(U),'vix valid',vz.notna().sum())
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avgN',round(a.n.mean(),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 if h==1:
  for yr,q in z.groupby(z.index.year): print('REG',yr,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
f.to_csv('scripts/miner_3_20261217_vix_level_reversal_signal.csv',index_label='date')
print('ARTIFACT scripts/miner_3_20261217_vix_level_reversal_signal.csv')
# correlation to unconditioned 3d reversal
z=pd.concat([f.stack().rename('f'),(-r3).stack().rename('base')],axis=1).dropna(); print('library_corr_base',z.f.corr(z.base))
