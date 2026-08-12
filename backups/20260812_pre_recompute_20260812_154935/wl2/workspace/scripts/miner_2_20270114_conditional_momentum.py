import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-01-13')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for a in A}).sort_index(); r=p.pct_change()
# Conditional factor: 20d momentum activated only when 5d return agrees with 20d trend; otherwise zero.
rows=[]; dates=[]; ns=[]; sig=[]
for i in range(25,len(r)-1):
    mom=r.iloc[i-1-19:i].sum(); short=r.iloc[i-1-4:i].sum()
    f=mom.where(np.sign(mom)==np.sign(short),0.0)
    y=r.iloc[i+1]; z=pd.concat([f,y],axis=1).dropna()
    if len(z)>=8:
        rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(r.index[i]); ns.append(len(z))
    sig.append((r.index[i],f))
ic=pd.Series(rows,index=dates)
print('dates',len(ic),'avgN',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
 z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
q=pd.DataFrame({d:f for d,f in sig}).T.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean().mean(),4),'coverage',round(q.notna().mean().mean(),4),'last',q.index[-1])
for n in [252,504]:
 z=ic.iloc[-n:]; print('recent',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('max_abs_library_correlation','not computed: no signal artifacts loaded')
