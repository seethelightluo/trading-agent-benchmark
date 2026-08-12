import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-18')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for a in A}).sort_index(); r=p.pct_change()
# Low-volatility signal, smoothed by blending 20d and 60d realized volatility; lagged one day.
fwd={1:[],3:[],5:[],10:[]}; ds={h:[] for h in fwd}; ns={h:[] for h in fwd}; sig=[]
for i in range(65,len(r)-10):
    v20=r.iloc[i-20:i].std(); v60=r.iloc[i-60:i].std(); f=-(0.65*v20+0.35*v60)
    sig.append((r.index[i],f))
    for h in fwd:
      y=r.iloc[i+1:i+1+h].sum(); z=pd.concat([f,y],axis=1).dropna()
      if len(z)>=8: fwd[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds[h].append(r.index[i]); ns[h].append(len(z))
for h in fwd:
 ic=pd.Series(fwd[h],index=ds[h]); print('H',h,'dates',len(ic),'avgN',np.mean(ns[h]),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
  z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
q=pd.DataFrame({d:f for d,f in sig}).T.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean().mean(),4),'coverage',round(r.loc[q.index].notna().mean().mean(),4),'last',q.index[-1])
for h in [1,3,5,10]:
 ic=pd.Series(fwd[h],index=ds[h]); z=ic.iloc[-252:]; print('recent',h,'252',round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
