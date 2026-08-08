import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(a):
    p=f'{base}/{a}.csv'
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date'])
    col='close'
    return d.set_index('date')[col].rename(a)
px=pd.concat([load(a) for a in assets],axis=1).sort_index()
# signal: recent 20d return minus annualized medium trend, scaled by trailing volatility
ret=px.pct_change()
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
# acceleration: short trend vs medium trend, volatility normalized; rank cross-section
sig=(px.pct_change(20)-px.pct_change(60)/3)/vol
# lag naturally uses close through t; evaluate t to t+h close return
out=[]
for h in [1,5,10,20]:
    fwd=px.shift(-h)/px-1
    rows=[]
    for dt in px.index:
        x=sig.loc[dt]; y=fwd.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    s=pd.Series(rows)
    out.append((h,len(s),s.mean(),s.std(ddof=1),s.mean()/s.std(ddof=1), (s>0).mean()))
print('UNIVERSE',len(assets),'DATES',len(px),'range',px.index.min().date(),px.index.max().date())
print('candidate=vol_normalized(20d_return-60d_return/3), cross-sectional Spearman IC')
for x in out: print('H%d dates=%d IC=%+.6f ICIR=%+.6f hit=%.3f'%(x[0],x[1],x[2],x[4],x[5]))
# rolling / regimes using H10
h=10; fwd=px.shift(-h)/px-1
ics=[]
for dt in px.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
ics=pd.Series(dict(ics)); print('REGIMES')
for name,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('latest120',None,None)]:
 s=ics if name!='latest120' else ics.tail(120)
 if name!='latest120': s=s.loc[a:b]
 print(name,'dates',len(s),'IC=%+.6f ICIR=%+.6f hit=%.3f'%(s.mean(),s.mean()/s.std(ddof=1), (s>0).mean()) if len(s)>2 else 'NA')
# signal coverage and turnover sampled every 10 sessions
print('coverage',sig.notna().mean().mean(),'active_dates',sig.notna().any(axis=1).sum())
r=sig.rank(axis=1,pct=True); sample=r.iloc[::10]
print('turnover10_proxy',sample.diff().abs().mean().mean())
print('decay',[(h,round(x[2],6)) for h,x in [(z[0],z) for z in out]])
# correlation evidence intentionally only against raw candidate proxies; library audit requires reconstruction
print('LIBRARY_AUDIT max_abs_library_correlation=UNAVAILABLE (existing JSON expressions not reconstructable by this script)')
