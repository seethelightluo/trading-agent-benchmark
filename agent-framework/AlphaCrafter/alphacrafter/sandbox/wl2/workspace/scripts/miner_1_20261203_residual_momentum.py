import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-11-17')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Residual medium-horizon momentum: asset 15d return relative to contemporaneous cross-sectional median,
# scaled by 40d volatility; removes common beta and avoids look-ahead.
rows={1:[],3:[],5:[],10:[]}; dates={h:[] for h in rows}; ns={h:[] for h in rows}; signals=[]
for i in range(45,len(r)-10):
    hist=r.iloc[:i]
    mom=r.iloc[i-15:i].sum(); cs= mom.median(); resid=mom-cs
    vol=r.iloc[i-40:i].std().replace(0,np.nan)
    f=resid/(0.002+np.sqrt(40)*vol)
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
