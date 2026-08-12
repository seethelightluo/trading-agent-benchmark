import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Volatility-normalized medium-term momentum, lagged one completed session.
series={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d)>100:
        x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        series[s]=x['close'].astype(float)
p=pd.DataFrame(series).sort_index().ffill()
r=np.log(p).diff()
# 20d return divided by 40d realized vol, with a mild consistency multiplier
mom=np.log(p/p.shift(20)); vol=r.rolling(40).std()*np.sqrt(40)
f=(mom/vol).replace([np.inf,-np.inf],np.nan)
# demean cross-section to remove common market drift
f=f.sub(f.mean(axis=1),axis=0)
fw=r.shift(-1)
rows=[]; sigrows=[]
for dt in f.index:
    z=f.loc[dt]; y=fw.loc[dt]; ok=z.notna()&y.notna()
    if ok.sum()>=8:
        ic=z[ok].corr(y[ok],method='spearman'); rows.append((dt,ic,ok.sum()))
        for s in U: sigrows.append((dt,s,float(z.get(s,np.nan))))
a=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
print('dates',len(a),'avg_n',a.n.mean(),'coverage',len(sigrows)/(len(f)*len(U)))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1), (a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
 q=a[(a.date.dt.year>=int(lo))&(a.date.dt.year<=int(hi))]
 print(lo+'-'+hi,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),4) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=np.log(p.shift(-h)/p); rr=[]
 for dt in f.index:
  z=f.loc[dt]; y=yy.loc[dt]; ok=z.notna()&y.notna()
  if ok.sum()>=8: rr.append(z[ok].corr(y[ok],method='spearman'))
 rr=pd.Series(rr).dropna(); print('horizon',h,'n',len(rr),'IC',rr.mean(),'ICIR',rr.mean()/rr.std(ddof=1))
# rank turnover
ranks=f.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
out=pd.DataFrame(sigrows,columns=['date','symbol','signal']); out.to_csv('scripts/miner_2_20290726_volnorm_momentum20_signal.csv',index=False)
