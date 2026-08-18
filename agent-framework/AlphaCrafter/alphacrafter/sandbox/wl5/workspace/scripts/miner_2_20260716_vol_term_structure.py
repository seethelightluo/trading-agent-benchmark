import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for a in A}).sort_index().ffill(); r=p.pct_change()
for w in [20,60]:
 f=-(r.rolling(5).std()/r.rolling(w).std()).replace([np.inf,-np.inf],np.nan)
 for h in [1,5,10]:
  vals=[]; counts=[]; dates=[]
  for i in range(w+5,len(r)-h):
   z=pd.DataFrame({'f':f.iloc[i], 'fr':r.iloc[i+1:i+h+1].sum()}).dropna()
   if len(z)>=8: vals.append(spearmanr(z.f,z.fr).statistic);counts.append(len(z));dates.append(r.index[i])
  s=pd.Series(vals,index=pd.to_datetime(dates));print('w',w,'h',h,'N',len(s),'mean_names',round(np.mean(counts),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4))
  if h==1: print('regimes',[(lab,round(s[(s.index.year>=lo)&(s.index.year<=hi)].mean(),5)) for lab,lo,hi in [('20-22',2020,2022),('23-24',2023,2024),('25-26',2025,2026)]])
 q=f.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).mean(),5),'coverage',round(f.notna().mean().mean(),4))
print('dates',len(r),'assets',len(A))
