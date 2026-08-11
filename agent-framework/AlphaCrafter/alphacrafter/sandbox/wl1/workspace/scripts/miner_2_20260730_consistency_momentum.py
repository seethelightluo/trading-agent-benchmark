import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,2500)
    if d is None or len(d)<100: d=get_index_daily_data(s,2500)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
print('assets',len(frames), 'lens', {k:len(v) for k,v in frames.items()})
px=pd.DataFrame(frames).sort_index().ffill()
# factor: medium-term trend persistence, return adjusted by vol and consistency of daily direction
ret=px.pct_change()
mom=px/px.shift(20)-1
vol=ret.rolling(20).std()*np.sqrt(20)
cons=ret.gt(0).rolling(20).mean()-0.5
fac=(mom/vol)*cons
fwd=px.shift(-10)/px-1
ics=[]; dates=[]; ranks=[]
for dt in fac.index:
    a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ranks.append(z.iloc[:,0].rank(pct=True))
ic=np.array(ics); ok=np.isfinite(ic)
ic=ic[ok]; dates=np.array(dates)[ok]
print('dates',len(ic),'mean_ic',ic.mean(),'std',ic.std(ddof=1),'icir',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
print('period',dates[0],dates[-1])
for yr in sorted(set(pd.to_datetime(dates).year)):
 q=ic[pd.to_datetime(dates).year==yr]
 print('year',yr,'n',len(q),'ic',q.mean(),'ir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# decay same sample for 1,5,10,20
for h in [1,5,10,20]:
 ff=px.shift(-h)/px-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 v=np.array(vals); v=v[np.isfinite(v)]
 print('decay',h,len(v),v.mean(),v.mean()/v.std(ddof=1))
# decision-step turnover based rank changes
rr=pd.DataFrame(ranks,index=dates).sort_index(); rr=rr.iloc[::10]
turn=(rr.diff().abs().mean(axis=1)).mean() if len(rr)>1 else np.nan
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover_rank10',turn)
