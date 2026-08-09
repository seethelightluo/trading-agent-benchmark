import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float).loc[:cut] for s in U}
P=pd.DataFrame(P).sort_index(); R=P.pct_change(fill_method=None)
# Cross-asset dispersion: rank each asset's 20d return relative to its own 120d history; reverse extreme recent winners.
r20=P.pct_change(20,fill_method=None); histmean=r20.rolling(120,min_periods=60).mean(); histsd=r20.rolling(120,min_periods=60).std(); F=-(r20-histmean)/histsd.replace(0,np.nan)
print('period',P.index.min().date(),P.index.max().date(),'assets',len(U),'rows',len(P))
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),6),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261217_relative_shock_signal.csv',index=False)
