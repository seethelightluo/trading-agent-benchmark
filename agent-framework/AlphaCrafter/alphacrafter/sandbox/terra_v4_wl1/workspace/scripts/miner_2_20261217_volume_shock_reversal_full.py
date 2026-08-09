import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2026-12-17')
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
V=pd.DataFrame({s:D[s].volume for s in U}).reindex(P.index)
# Candidate: volume-shock reversal, winsorized cross-sectionally to limit crypto/roll artifacts
sur=V.div(V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
raw=-(P/P.shift(3)-1)*np.log1p(sur.shift(1).clip(lower=0))
f=raw.clip(lower=raw.quantile(.05,axis=1),upper=raw.quantile(.95,axis=1),axis=0)
f.to_csv('scripts/miner_2_20261217_volume_shock_reversal_full_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),8),'ICIR',round(ic.mean()/ic.std(ddof=1),8),'hit',round((ic>0).mean(),4))
 if h==1:
  for regime,g in ic.groupby(ic.index.year): print('YR',regime,len(g),round(g.mean(),6),round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'period',P.index.min().date(),P.index.max().date())
print('artifact rows',len(f),'cols',len(f.columns))
