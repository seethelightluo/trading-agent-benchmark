import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-09-24')
px={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d[d.date<=cut].drop_duplicates('date').set_index('date').close
        px[s]=d
P=pd.DataFrame(px).sort_index()
r=P.pct_change()
# candidate: five-day bounded reversal normalized by 20d vol, gated by elevated cross-sectional dispersion
ret5=P/P.shift(5)-1
vol20=r.rolling(20).std()*np.sqrt(252)
disp=r.sub(r.median(axis=1),axis=0).abs().median(axis=1).rolling(20).mean()
# use only information through t; signal at t predicts t+1
f=-(ret5/(1+ret5.abs()))/vol20
# conditional amplification in high dispersion, interpretable and bounded
threshold=disp.rolling(252,min_periods=80).median()
f=f.where(disp.ge(threshold), f*0.35)
ics=[]; dates=[]; turnovers=[]; cov=[]
for i in range(len(P)-1):
    x=f.iloc[i]; y=r.iloc[i+1]
    ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ics.append(x[ok].corr(y[ok],method='spearman')); dates.append(P.index[i]); cov.append(ok.mean())
        if i>0:
            a=f.iloc[i-1]; b=x; z=a.notna()&b.notna()
            if z.sum()>=8: turnovers.append((a[z].rank().corr(b[z].rank()) is not None))
ics=np.array(ics,dtype=float); ics=ics[np.isfinite(ics)]
print('dates',len(ics),'avg instruments',np.mean([((f.loc[d].notna()&r.loc[d+pd.Timedelta(days=1)].notna()).sum()) for d in dates if d+pd.Timedelta(days=1) in r.index]))
print('coverage',np.mean(cov),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1)*np.sqrt(252),'hit',np.mean(ics>0))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 z=ics[(np.array(dates)>=pd.Timestamp(a+'-01-01'))&(np.array(dates)<=pd.Timestamp(b+'-12-31'))]
 print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
# horizons
for h in [1,5,10]:
 vals=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(x[ok].corr(y[ok],method='spearman'))
 vals=np.array(vals); print('h',h,'n',len(vals),'ic',np.nanmean(vals),'icir',np.nanmean(vals)/np.nanstd(vals,ddof=1)*np.sqrt(252))
# write artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280925_dispersion_conditional_5d_reversal_signal.csv',index=False)
