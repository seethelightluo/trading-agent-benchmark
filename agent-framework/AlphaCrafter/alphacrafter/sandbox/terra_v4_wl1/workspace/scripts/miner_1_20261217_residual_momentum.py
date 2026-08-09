import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']).dt.normalize(); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# Candidate: residual 20d momentum scaled by trailing 20d vol; residual to contemporaneous cross-sectional median
m20=p.pct_change(20); med=m20.median(axis=1); resid=m20.sub(med,axis=0)
vol=r.rolling(20).std(); f=resid/(vol*np.sqrt(20))
# use signal at t, forward returns t+1,5,10; rank IC
rows=[]
for d in f.index:
    vals=f.loc[d]
    rec={'date':d,'n':int(vals.notna().sum())}
    for h in [1,5,10]:
        fr=p.pct_change(h).shift(-h).loc[d]
        z=pd.concat([vals,fr],axis=1).dropna()
        rec[f'ic{h}']=z.iloc[:,0].rank().corr(z.iloc[:,1].rank()) if len(z)>=8 else np.nan
    rows.append(rec)
a=pd.DataFrame(rows).set_index('date').dropna(subset=['ic1'])
print('candidate residual risk-adjusted 20d momentum; dates',len(a),'avg_names',a.n.mean(),'range',a.index.min(),a.index.max())
for h in [1,5,10]:
    q=a[f'ic{h}'].dropna(); print(h,'IC %.5f ICIR %.5f hit %.3f n=%d'%(q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for y,g in a.groupby(a.index.year):
 q=g.ic1.dropna(); print('year',y,'ic %.5f icir %.5f n=%d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
# regime halves
for label,g in [('early',a[a.index<'2023-01-01']),('late',a[a.index>='2023-01-01']),('recent',a[a.index>='2025-01-01'])]:
 q=g.ic1.dropna(); print(label,'ic %.5f icir %.5f hit %.3f n=%d'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
print('sample',f.tail(1).to_dict('records'))
