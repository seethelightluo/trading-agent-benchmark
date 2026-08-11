import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2028-05-17'))
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Drawdown-recovery balance: assets nearer recovered 60d lows and with positive medium return rank higher.
lo=C.rolling(60,min_periods=45).min(); hi=C.rolling(60,min_periods=45).max(); v=r.rolling(20,min_periods=15).std()
recovery=(C/lo-1); drawdown=(hi/C-1)
raw=(0.6*(recovery-drawdown)+0.4*(C/C.shift(20)-1))/v.replace(0,np.nan)
F=raw.shift(1).clip(-20,20)
print('end',end.date())
for h in [1,3,5,10]:
 y=C.shift(-h).div(C)-1; a=[]; ds=[]; ns=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): a.append(q);ds.append(dt);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo_y,hi_y in [(2020,2022),(2023,2025),(2026,2027),(2028,2028)]:
  z=a[[lo_y<=d.year<=hi_y for d in ds]]; print('regime',lo_y,hi_y,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
