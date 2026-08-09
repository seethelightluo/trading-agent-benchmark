import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(P).sort_index(); p=p.loc[:'2027-03-24']; r=p.pct_change()
# Low-volatility defensive signal: negative 30-day realized volatility.
fac=-r.rolling(30,min_periods=20).std()
for h in [1,5,10]:
 fwd=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): vals.append((dt,ic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('h',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 if h==1:
  for name,a,b in [('20-22','2020','2022'),('23-24','2023','2024'),('25-27','2025','2027')]:
   x=q.loc[a:b]; print(name,len(x),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6))
print('coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_lowvol30_signal.csv',index=False)
