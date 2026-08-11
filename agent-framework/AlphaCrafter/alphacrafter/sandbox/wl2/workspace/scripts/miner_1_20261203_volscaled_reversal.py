import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-17')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date')['close'] for a in A}; p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Volatility-scaled short reversal, with a stable 20d trend gate: reverse recent shocks only when
# the 20d trend is not strongly adverse. All inputs end at decision date.
res={h:[] for h in [1,3,5,10]}; ds={h:[] for h in res}; ns={h:[] for h in res}; sig=[]
for i in range(45,len(r)-10):
    x=r.iloc[i-3:i].sum(); trend=r.iloc[i-20:i].sum(); vol=r.iloc[i-40:i].std();
    f=-x/(.002+np.sqrt(40)*vol)
    f=f.where(trend>-0.10, f*0.25) # damp, rather than discard, persistent losers
    sig.append((r.index[i],f))
    for h in res:
      y=r.iloc[i+1:i+1+h].sum(); z=pd.concat([f,y],axis=1).dropna()
      if len(z)>=8: res[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds[h].append(r.index[i]); ns[h].append(len(z))
for h in res:
 ic=pd.Series(res[h],index=ds[h]); print('H',h,'dates',len(ic),'avgN',round(np.mean(ns[h]),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
  q=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
q=pd.DataFrame({d:f for d,f in sig}).T.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean().mean(),4),'coverage',round(r.loc[q.index].notna().mean().mean(),4))
for h in res:
 ic=pd.Series(res[h],index=ds[h]); z=ic.iloc[-252:]; print('recent',h,'252',round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('max_abs_library_correlation unavailable')
