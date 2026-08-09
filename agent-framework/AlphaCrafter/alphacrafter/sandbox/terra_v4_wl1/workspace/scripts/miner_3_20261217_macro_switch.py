import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
V=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].reindex(P.index).ffill()
D=pd.read_csv(f'{macro}/DXY.csv',parse_dates=['date']).set_index('date').sort_index()['close'].reindex(P.index).ffill()
# All signals use completed-day lagged data. Macro-conditioned defensive rotation: reverse short-term moves in calm regimes, but favor relative strength in stressed regimes.
r3=P.pct_change(3); vixrank=V.rolling(252,min_periods=60).rank(pct=True); dtrend=D.pct_change(10)
variants={
 'vix_switch': (-r3).where(vixrank<.70, r3),
 'dxy_switch': (-r3).where(dtrend<0, r3),
 'joint_switch': (-r3).where((vixrank<.70)&(dtrend<0), r3)
}
Y={h:P.shift(-h).div(P)-1 for h in [1,5,10]}
for name,f0 in variants.items():
 f=f0.shift(1) # explicit lag
 print('\n',name)
 for h,y in Y.items():
  rows=[]
  for d in P.index:
   q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
  a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
  print(h,len(ic),round(a.n.mean(),2),round(ic.mean(),6),round(ic.mean()/ic.std(ddof=1),6),round((ic>0).mean(),4))
 print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
 f.to_csv(f'scripts/miner_3_20261217_{name}_signal.csv',index_label='date')
 print('artifact',f'scripts/miner_3_20261217_{name}_signal.csv')
