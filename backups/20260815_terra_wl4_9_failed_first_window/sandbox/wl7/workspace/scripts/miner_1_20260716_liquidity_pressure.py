import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
d={}
for s in U:
    x=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
    d[s]=x
# liquidity-weighted signed pressure: CLV times abnormal volume, smoothed 10d
# volume is cross-asset normalized by own 60d median, avoiding price scale
frames=[]
for s,x in d.items():
    clv=((x.high-x.low).replace(0,np.nan))
    clv=(2*x.close-x.high-x.low)/clv
    vz=x.volume/(x.volume.rolling(60,min_periods=30).median())-1
    pressure=(clv*vz).rolling(10,min_periods=8).mean()
    ret=x.close.pct_change()
    frames.append(pd.DataFrame({'date':x.index,'asset':s,'f':pressure,'r1':ret.shift(-1),'r5':x.close.pct_change(5).shift(-5),'r10':x.close.pct_change(10).shift(-10)}))
z=pd.concat(frames,ignore_index=True).dropna(subset=['f','r1'])
rows=[]
for dt,g in z.groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.r1.nunique()>1:
        q={'date':dt,'n':len(g)}
        for h in ['r1','r5','r10']:
            a=g.dropna(subset=[h]); q[h]=spearmanr(a.f,a[h]).statistic if len(a)>=8 and a.f.nunique()>1 and a[h].nunique()>1 else np.nan
        rows.append(q)
r=pd.DataFrame(rows)
print('dates',len(r),'mean_n',r.n.mean(),'coverage',len(z)/sum(len(x) for x in d.values()))
for h in ['r1','r5','r10']:
    a=r[h].dropna(); print(h,'IC %.5f ICIR %.5f hit %.4f obs %d'%(a.mean(),a.mean()/a.std(),(a>0).mean(),len(a)))
for yr in [(2020,2022),(2023,2024),(2025,2026)]:
 a=r[(r.date.dt.year>=yr[0])&(r.date.dt.year<=yr[1])]['r1'].dropna();print('regime',yr,'n',len(a),'ICIR',a.mean()/a.std(),'IC',a.mean())
# average daily rank turnover
p=z.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',p.diff().abs().mean().mean(),'cross-sectional std',p.std(axis=1).mean())
# pooled correlation with known approximate factors
for h in [5,20]:
 mom=pd.concat([pd.DataFrame({'date':x.index,'asset':s,'m':x.close.pct_change(h)}) for s,x in d.items()],ignore_index=True)
 q=z[['date','asset','f']].merge(mom,on=['date','asset']).dropna();print('corr mom',h,q.f.corr(q.m,method='spearman'))
print('recent',r.tail(300).r1.mean(),r.tail(300).r1.mean()/r.tail(300).r1.std(),len(r.tail(300)))
