import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
# downside-asymmetry momentum: 20d return divided by downside deviation, lagged one day
r=p.pct_change()
ret20=p.pct_change(20)
down=r.where(r<0,0).rolling(20,min_periods=12).std()
f=(ret20/(down*np.sqrt(20)+1e-8)).shift(1)
# forward 10 trading-day simple return
fwd=p.shift(-10)/p-1
rows=[]; dates=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if np.isfinite(ic): rows.append(ic); dates.append(dt)
ic=np.array(rows)
# rolling signal turnover: rank changes across consecutive observations on common universe
turn=[]; prev=None
for dt in f.index:
    z=f.loc[dt].dropna()
    if len(z)<8: continue
    q=z.rank(pct=True)
    if prev is not None:
        common=q.index.intersection(prev.index)
        if len(common)>=8: turn.append(np.mean(np.abs(q[common]-prev[common])))
    prev=q
print('dates',len(ic),'instruments_mean',round(np.mean([f.loc[d].notna().sum() for d in dates]),2),'coverage',round(np.mean([f.loc[d].notna().mean() for d in dates]),4))
print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/(ic.std(ddof=1)+1e-12)*np.sqrt(len(ic)),6),'hit',round(np.mean(ic>0),4),'turnover',round(np.mean(turn),6))
for n in [250,500]:
 q=ic[-n:] if len(ic)>=n else ic
 print('recent',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)),6))
for h in [5,10,20]:
 fw=p.shift(-h)/p-1; v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): v.append(c)
 print('horizon',h,'dates',len(v),'IC',round(np.mean(v),6),'ICIR',round(np.mean(v)/(np.std(v,ddof=1)+1e-12)*np.sqrt(len(v)),6))
