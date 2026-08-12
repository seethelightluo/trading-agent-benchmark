import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-11-04')
data={}
for a in assets:
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')
    data[a]=x
close=pd.DataFrame({a:x['close'] for a,x in data.items()}).sort_index(); ret=close.pct_change()
# Range-adjusted persistence: 30d return / aggregate true range, gated by positive-day fraction.
tr=pd.DataFrame({a:(x['high']-x['low']).abs()/x['close'].shift(1) for a,x in data.items()}).reindex(close.index)
ics=[]; ds=[]; ns=[]; signals=[]
for i in range(65,len(close)-10):
    w=ret.iloc[i-30:i]; tw=tr.iloc[i-30:i]
    f=w.sum()/(tw.sum()+0.05) * (w.gt(0).mean()**1.5)
    y=ret.iloc[i+1:i+11].sum(); z=pd.concat([f,y],axis=1).dropna()
    if len(z)>=8:
        ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(close.index[i]); ns.append(len(z)); signals.append((close.index[i],f))
ic=pd.Series(ics,index=ds)
print('dates',len(ic),'avgN',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
 z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
q=pd.DataFrame({d:f for d,f in signals}).T.rank(axis=1,pct=True)
print('turnover',round(q.diff().abs().mean().mean(),4),'coverage',round(close.loc[q.index].notna().mean().mean(),4),'last',q.index[-1])
for n in [63,126,252,504]:
 z=ic.iloc[-n:]; print('recent',n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [1,3,5,20]:
 vals=[]; dates=[]
 for i in range(65,len(close)-h):
  w=ret.iloc[i-30:i];tw=tr.iloc[i-30:i];f=w.sum()/(tw.sum()+.05)*(w.gt(0).mean()**1.5); y=ret.iloc[i+1:i+1+h].sum();z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(close.index[i])
 zz=pd.Series(vals,index=dates);print('decay',h,round(zz.mean(),6),round(zz.mean()/zz.std(ddof=1),6))
print('max_abs_library_correlation unavailable: candidate artifact not persisted')
