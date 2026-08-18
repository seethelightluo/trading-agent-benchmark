import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volatility-normalized trend curvature, short trend relative to long trend.
# Lag all inputs one completed day; rank IC against 10-session forward returns.
px={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
P=pd.DataFrame(px).sort_index().ffill()
r=np.log(P).diff()
# curvature: recent 20d return minus 1/3 of 60d return, normalized by trailing 20d vol
f=(P.shift(1)/P.shift(21)-1 - (P.shift(1)/P.shift(61)-1)/3) / (r.shift(1).rolling(20).std()*np.sqrt(252)+1e-12)
y=P.shift(-10)/P-1
ics=[]; nms=[]; turnover=[]
prev=None
for dt in f.index:
    a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        ics.append(a[ok].corr(b[ok],method='spearman')); nms.append(ok.sum())
    z=a.rank(pct=True)
    if prev is not None:
        q=z.notna()&prev.notna(); turnover.append((z[q]-prev[q]).abs().mean())
    prev=z
ic=pd.Series(ics).dropna(); arr=ic.values
f.to_csv('factors/miner_1_20350216_volnorm_curvature_20_60_signal.csv')
print('candidate=volnorm_curvature_20_60 dates',len(ic),'avg_names',round(float(np.mean(nms)),3),'coverage',round(float(np.mean(nms)/15),4))
print('IC',round(float(ic.mean()),6),'ICIR',round(float(ic.mean()/ic.std(ddof=1)),6),'hit',round(float((ic>0).mean()),4),'turnover',round(float(np.mean(turnover)),4))
for k in [120,252,504]:
 z=ic.tail(k); print('recent',k,'n',len(z),'ICIR',round(float(z.mean()/z.std(ddof=1)),6),'IC',round(float(z.mean()),6))
# block ICIR
for i,z in enumerate(np.array_split(ic,4)): print('block',i+1,'n',len(z),'IC',round(float(z.mean()),6),'ICIR',round(float(z.mean()/z.std(ddof=1)),6))
print('decay')
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; vv=[]
 for dt in f.index:
  a=f.loc[dt]; b=yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vv.append(a[ok].corr(b[ok],method='spearman'))
 print(h,round(float(pd.Series(vv).mean()),6),len(vv))
