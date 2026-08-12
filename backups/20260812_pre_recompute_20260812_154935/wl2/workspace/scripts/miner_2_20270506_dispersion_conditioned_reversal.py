import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=pd.Timestamp('2027-05-05')
dates=D['SPX'].index[(D['SPX'].index>='2020-03-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Dispersion-conditioned short-term reversal: reversal is stronger when recent
# cross-asset return dispersion is elevated; all inputs lagged one session.
disp=R.rolling(20,min_periods=15).std().mean(axis=1)
F=(-C.pct_change(3).mul(disp,axis=0)).shift(1)
Y={h:C.shift(-h).div(C)-1 for h in [1,3,5,10]}
def run(y):
  ic=[]; ds=[]; ns=[]
  for dt in dates:
   z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8:
    q=spearmanr(z.f,z.y).statistic
    if np.isfinite(q):ic.append(q);ds.append(dt);ns.append(len(z))
  a=np.array(ic); print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
  for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
   z=a[[lo<=d.year<=hi for d in ds]]
   print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
for h,y in Y.items(): print('HORIZON',h);run(y)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'end',end.date())
# Conditional comparison: only high-dispersion dates (above expanding median, lagged)
cut=disp.shift(1).expanding(min_periods=100).median(); mask=disp.shift(1)>cut
for h,y in [(1,Y[1]),(5,Y[5])]:
 a=[]
 for dt in dates[mask.reindex(dates).fillna(False)]:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 a=np.array(a);print('COND highdisp h',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
