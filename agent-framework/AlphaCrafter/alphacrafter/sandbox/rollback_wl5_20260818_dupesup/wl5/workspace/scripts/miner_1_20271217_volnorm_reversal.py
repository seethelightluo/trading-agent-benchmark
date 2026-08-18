import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
CUT=pd.Timestamp('2027-12-16')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
    f='../persistent/stock_data/'+s+'.csv'
    d=pd.read_csv(f)
    d['date']=pd.to_datetime(d['date']); d=d[d.date<=CUT].sort_values('date').set_index('date')
    px[s]=d.close.astype(float)
P=pd.concat(px,axis=1).sort_index()
r=P.pct_change()
# Candidate: volatility-normalized 5-day reversal; signal at t, forward 10d return t+1..t+10
sig=-(P/P.shift(5)-1)/(r.rolling(20).std()*np.sqrt(252)+1e-8)
fwd=P.shift(-10)/P.shift(-1)-1
ics=[]; dates=[]; nobs=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); nobs.append(len(z))
ic=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
print('cutoff',CUT.date(),'dates',len(ic),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12), (ic>0).mean()))
for a,b in [('2020-2022','2022-12-31'),('2023-2024','2024-12-31'),('2025-2026','2026-12-31'),('2027','2027-12-16')]:
    st=a[:4]+'-01-01'; q=ic.loc[st:b]; print(a,len(q),q.mean() if len(q) else np.nan)
for h in [1,5,10,20]:
    fw=P.shift(-h)/P.shift(-1)-1; vals=[]
    for dt in sig.index:
      z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    print('decay',h,np.nanmean(vals),len(vals))
# approximate turnover rank signal
rank=sig.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
print('last',sig.loc[:CUT].tail(1).T.to_string(header=False))
