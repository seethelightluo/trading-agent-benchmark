import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: medium horizon return divided by downside volatility, a defensive relative-strength signal
px={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None: d=get_index_daily_data(s, days=4000)
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change()
# factor at t uses through t, then explicitly lag one session for forward testing
ret20=P/P.shift(20)-1
down=r.where(r<0,0).rolling(40).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
f=ret20/(down+0.005)
f=f.shift(1)
# forward non-overlapping definition from each date to t+h
out=[]
for h in [5,10,20]:
  vals=[]; dates=[]; ns=[]
  for i in range(len(P)-h):
    x=f.iloc[i]; y=(P.iloc[i+h]/P.iloc[i]-1)
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(P.index[i]); ns.append(len(z))
  a=np.asarray(vals); mu=np.nanmean(a); sd=np.nanstd(a,ddof=1)
  print(h, 'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',mu,'ICIR',mu/sd,'hit',np.mean(a>0))
# rank turnover
ranks=f.rank(axis=1,pct=True); tv=(ranks-ranks.shift(1)).abs().mean(axis=1).mean()
print('turnover',tv,'assets',len(px),'dates',len(P))
# regime recent diagnostics
for label, start in [('2026', '2026-01-01'),('2027','2027-01-01')]:
  h=20; vals=[]
  for i in range(len(P)-h):
   if P.index[i]<pd.Timestamp(start): continue
   z=pd.concat([f.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  a=np.asarray(vals); print(label,'n',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1))
# save signal artifact
f.to_csv('scripts/miner_1_20271118_downside_relative_strength_signal.csv')
