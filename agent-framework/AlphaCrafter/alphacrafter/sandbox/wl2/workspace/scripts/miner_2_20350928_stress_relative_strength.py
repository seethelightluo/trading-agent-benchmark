import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: lagged 20d risk-adjusted momentum, activated by broad cross-asset stress
# Stress = breadth of negative 10d returns >= 60%; score favors relative strength in stress.
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index(); ret=p.pct_change();
# signal available at t uses through t-1
r10= p.pct_change(10); r20=p.pct_change(20); vol20=ret.rolling(20).std()
breadth=(r10<0).mean(axis=1)
active=(breadth>=.60)
# relative risk-adjusted trend; cross-sectional centering makes scale comparable
raw=r20/vol20.replace(0,np.nan); sig=raw.sub(raw.median(axis=1),axis=0).where(active, np.nan).shift(1)
fwd=p.shift(-10)/p-1
rows=[]; active_dates=0
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
        rows.append((dt,ic,len(z)))
        if active.loc[dt]: active_dates+=1
res=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
# only dates with active signal are intended observations
res=res[active.reindex(res.date).fillna(False).to_numpy()]
for h in [1,5,10,20]:
    fy=p.shift(-h)/p-1; rr=[]
    for dt in sig.index:
      if not active.get(dt,False): continue
      z=pd.concat([sig.loc[dt],fy.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(rr).dropna(); print('H',h,'dates',len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1), (a>0).mean()))
print('all dates',len(res),'avgN',res.n.mean(),'active_frac',active.mean(),'coverage_active',res.n.mean()/len(U))
print('periods')
for lo,hi in [('2020','2025'),('2026','2028'),('2029','2032'),('2033','2035')]:
 a=res[(res.date>=lo)&(res.date<=hi)].ic
 print(lo,hi,len(a),a.mean() if len(a) else np.nan,(a.mean()/a.std(ddof=1)) if len(a)>1 else np.nan)
# turnover on active consecutive observations: rank correlation between adjacent signals
turn=[]
for i in range(1,len(sig)):
 if active.iloc[i] and active.iloc[i-1]:
  a=sig.iloc[i]; b=sig.iloc[i-1]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover_proxy',np.nanmean(turn) if turn else np.nan,'n',len(turn))
