import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    z=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close']
    P[s]=z
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
med=r.median(axis=1); disp=r.sub(med,axis=0).abs().median(axis=1)
# Candidate: 3d residual reversal, inverse 20d volatility, activated only in elevated AND rising dispersion.
res3=r.rolling(3,min_periods=3).sum().sub(r.rolling(3,min_periods=3).sum().median(axis=1),axis=0)
vol=r.rolling(20,min_periods=10).std()
activation=(disp>disp.rolling(60,min_periods=30).median()) & (disp>disp.shift(5))
sig=(-res3/vol).where(activation, np.nan)
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for dt in sig.index:
    q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
        rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=dispersion_rising_volnorm_residual_reversal')
print('dates',len(d),'avgN',round(d.n.mean(),3),'IC',round(d.ic.mean(),6),'ICIR',round(d.ic.mean()/d.ic.std(ddof=1),6),'hit',round((d.ic>0).mean(),4),'coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
for label,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025','2025','2025'),('2026+','2026-07','2027')]:
 q=d.loc[a:b].ic; print(label,'dates',len(q),'IC',round(q.mean(),6) if len(q) else np.nan,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
out=sig.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_dispersion_rising_residual.csv',index=False)
print('artifact_rows',len(out))
