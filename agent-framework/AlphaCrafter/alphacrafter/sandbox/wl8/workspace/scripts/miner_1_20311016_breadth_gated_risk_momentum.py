import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
p=pd.concat(frames,axis=1).sort_index(); r=p.pct_change()
# signal at t: risk-adjusted intermediate trend gated by contemporaneous cross-asset breadth
ret=p.pct_change(20); vol=r.rolling(20).std()*np.sqrt(252)
breadth=(ret>0).mean(axis=1)
f=(ret/vol)*(0.5+breadth) # smooth positive breadth gate, no future information
# one day lag before forward returns
sig=f.shift(1); fw=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
    a=sig.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
        rows.append((dt,ic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
q=q.replace([np.inf,-np.inf],np.nan).dropna()
mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('dates',len(q),'period',q.index.min().date(),q.index.max().date(),'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
print('IC',mean,'daily_ICIR',mean/sd if sd else np.nan,'annualized_ICIR',icir,'hit',(q.ic>0).mean())
for n in [60,180,365,756]:
 z=q.tail(n); print('recent',n,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'dates',len(z))
print('year')
print(q.groupby(q.index.year).ic.agg(['mean','count']).tail(8).to_string())
# turnover: rank top/bottom changes, use mean absolute signal change normalized
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
# decay
for h in [1,5,10,20]:
 fw2=p.shift(-h)/p-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw2.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,np.nanmean(vals),len(vals))
# artifacts for provenance
sig.to_csv('scripts/miner_1_20311016_breadth_gated_risk_momentum_signal.csv')
q.to_csv('scripts/miner_1_20311016_breadth_gated_risk_momentum_ic.csv')
