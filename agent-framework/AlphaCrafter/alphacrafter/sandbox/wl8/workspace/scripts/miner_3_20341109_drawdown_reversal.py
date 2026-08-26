import numpy as np, pandas as pd
import os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    f='../persistent/stock_data/'+s+'.csv'
    if os.path.exists(f):
        x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# Candidate: normalized distance below 60d high, conditioned on positive 120d trend.
# Contrarian score: drawdown magnitude / 20d vol, only when long trend positive; otherwise neutral.
high=p.rolling(60,min_periods=45).max(); vol=r.rolling(20,min_periods=15).std()
trend=p.pct_change(120); raw=-(1-p/high)/(vol*np.sqrt(20))
raw=raw.where(trend>0)
# lag signal explicitly, forward 10d return
sig=raw.shift(1); fwd=p.shift(-10)/p-1
rows=[]; date_sigs=[]
for dt in sig.index:
    a=sig.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
        date_sigs.append(a.notna().mean())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# use dates with 2020 onward and exclude last 10 unavailable
ic=ic.loc['2020-01-01':].dropna(); vals=ic.ic
mean=vals.mean(); sd=vals.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
# turnover based rank direction/signal changes among overlapping names
ranks=sig.rank(axis=1,pct=True); turnovers=[]
for i in range(1,len(ranks)):
    a=ranks.iloc[i-1]; b=ranks.iloc[i]; q=pd.concat([a,b],axis=1).dropna()
    if len(q)>=8: turnovers.append((q.iloc[:,1]-q.iloc[:,0]).abs().mean())
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',np.mean(date_sigs),'IC',mean,'ICIR',icir,'hit',np.mean(vals>0),'turnover',np.mean(turnovers))
for w in [365,750,1260]:
 print('recent',w, vals.tail(w).mean()/vals.tail(w).std(ddof=1)*np.sqrt(252))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr))
# artifacts
out=sig.loc[ic.index].stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20341109_drawdown_reversal_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_3_20341109_drawdown_reversal_ic.csv',index=False)
