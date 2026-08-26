import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: medium-term risk-adjusted momentum, activated only when cross-asset breadth is healthy.
# This tests whether trend continuation is more reliable outside broad stress.
raw={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<250: d=get_index_daily_data(s, days=4000)
    if d is not None:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
        raw[s]=x.close.astype(float)
px=pd.concat(raw,axis=1).sort_index().ffill()
r=px.pct_change()
# breadth signal uses only information at t; factor is not defined on weak breadth days
breadth=(r.rolling(20).mean()>0).mean(axis=1)
vol=r.rolling(20).std()*np.sqrt(252)
ret=px.pct_change(20)
# consistency is fraction positive daily returns in prior 20d; interpretable
cons=(r>0).rolling(20).mean()
f=(ret/(vol.replace(0,np.nan)))*((cons-0.5)*2)
f=f.where(breadth>=0.50)
# forward 10 trading day return, calculate date-wise cross-sectional rank IC
fr=px.shift(-10)/px-1
rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=breadth_conditioned_risk_adjusted_momentum_10d')
print('dates',len(out),'mean_n',out.n.mean(),'coverage',out.n.mean()/15)
for name, sub in [('full',out),('2020_2023',out.loc[:'2023-12-31']),('2024_2026',out.loc['2024-01-01':'2026-12-31']),('2027_2028',out.loc['2027-01-01':'2028-12-31']),('2029',out.loc['2029-01-01':]),('recent252',out.tail(252))]:
    if len(sub): print(name,'n',len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(len(sub)) if sub.ic.std(ddof=1)>0 else np.nan,'hit',(sub.ic>0).mean())
# decay
for h in [5,10,20]:
    fh=px.shift(-h)/px-1; rr=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
        if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
    a=pd.Series(rr).dropna(); print('decay',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)))
# naive turnover: rank ordering changes on adjacent valid dates
rank=f.rank(axis=1,pct=True); print('mean_abs_rank_change',rank.diff().abs().mean(axis=1).mean())
