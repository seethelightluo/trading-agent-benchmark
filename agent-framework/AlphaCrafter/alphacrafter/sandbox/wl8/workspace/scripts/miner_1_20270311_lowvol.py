import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2027-03-10'
r={}
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 r[s]=d.close.pct_change()
r=pd.DataFrame(r); f=-r.rolling(20,min_periods=15).std(); forward=r.shift(-1)
ics=[]; n=[]; dates=[]; turns=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt].rename('f'),forward.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.f,z.y).statistic);n.append(len(z));dates.append(dt)
 q=f.loc[dt].rank(pct=True)
 if prev is not None:turns.append((q-prev).abs().mean())
 prev=q
x=np.array(ics);print('dates',len(x),'avg_names',np.mean(n),'coverage',np.mean(n)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turnover',np.mean(turns))
for start,end in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 q=x[[start<=d.year<=end for d in dates]];print('regime',start,end,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
