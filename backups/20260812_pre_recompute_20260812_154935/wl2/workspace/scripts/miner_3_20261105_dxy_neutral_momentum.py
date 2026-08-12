import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-11-04')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'].reindex(r.index).ffill().pct_change()
rows={1:[],3:[],5:[],10:[]}; dates={h:[] for h in rows}; ns={h:[] for h in rows}; signals=[]
for i in range(65,len(r)-10):
    rr=r.iloc[i-40:i]; m=macro.iloc[i-40:i]
    if m.notna().sum()<30: continue
    var=m.var()
    beta=rr.apply(lambda x: x.cov(m)/var if var>1e-12 else np.nan)
    mom=r.iloc[i-20:i].sum(); dm=m.iloc[-20:].sum()
    f=mom-beta*dm
    signals.append((r.index[i],f))
    for h in rows:
        y=r.iloc[i+1:i+1+h].sum(); z=pd.concat([f,y],axis=1).dropna()
        if len(z)>=8:
            rows[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates[h].append(r.index[i]); ns[h].append(len(z))
for h in rows:
    ic=pd.Series(rows[h],index=dates[h]); print('H',h,'dates',len(ic),'avgN',round(np.mean(ns[h]),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
    for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2026)]:
        z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
q=pd.DataFrame({d:f for d,f in signals}).T.rank(axis=1,pct=True)
print('turnover',round(q.diff().abs().mean().mean(),4),'coverage',round(r.loc[q.index].notna().mean().mean(),4),'signal_dates',len(q),'last',q.index[-1])
for h in [1,3,5,10]:
    ic=pd.Series(rows[h],index=dates[h]);
    for n in [63,126,252]:
        z=ic.iloc[-n:]; print('recent',h,n,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('max_abs_library_correlation unavailable: candidate artifact not persisted')
