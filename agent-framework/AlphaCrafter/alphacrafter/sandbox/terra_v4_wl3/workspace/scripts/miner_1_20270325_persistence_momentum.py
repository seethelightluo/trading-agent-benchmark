import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); P[s]=d.close.astype(float)
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); hit=(r.rolling(30).mean()/(r.abs().rolling(30).mean()+1e-9)).clip(-1,1); fac=p.pct_change(30)*hit/(r.rolling(30).std()*np.sqrt(30)+1e-9)
for h in [1,5,10]:
 fwd=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): vals.append((dt,ic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); print('h',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit', (q.ic>0).mean())
 if h==1:
  for name,a,b in [('20-22','2020','2022'),('23-24','2023','2024'),('25-27','2025','2027')]:
   x=q.loc[a:b]; print(name,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 else np.nan)
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean()); fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_persistence_momentum_signal.csv',index=False)
