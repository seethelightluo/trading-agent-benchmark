import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
acct=get_account_dict(); uni=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in uni:
    d=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: d=fn(s,days=3000)
        except Exception: pass
        if d is not None and len(d): break
    if d is not None and len(d):
        x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); frames[s]=x.dropna().drop_duplicates('date').set_index('date').close.sort_index()
px=pd.DataFrame(frames).sort_index().ffill(); r=px.pct_change()
# Intermediate-term momentum: 252 sessions return, excluding most recent 5 sessions, lagged one day
sig=(px.shift(5)/px.shift(252)-1).shift(1)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): vals.append(q); dates.append(dt); ns.append(len(z))
 v=np.array(vals); ir=v.mean()/v.std(ddof=1)*np.sqrt(len(v))
 print('H',h,'dates',len(v),'avg_n',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(ir,6),'hit',round((v>0).mean(),4))
 if h==10:
  rank=sig.rank(axis=1,pct=True); t=[]
  for i in range(1,len(rank)):
   c=rank.iloc[i].dropna().index.intersection(rank.iloc[i-1].dropna().index)
   if len(c)>=8:t.append(np.abs(rank.iloc[i][c]-rank.iloc[i-1][c]).mean())
  print('TURN',round(np.mean(t),6),'COVERAGE',round(sig.notna().sum().sum()/(sig.shape[0]*len(uni)),4),'assets',len(frames))
  for lab,a,b in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31'),('2028','2028','2028-12-31')]:
   q=np.array([vals[i] for i,d in enumerate(dates) if pd.Timestamp(a)<=d<=pd.Timestamp(b)])
   print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6) if len(q)>1 and q.std(ddof=1)>0 else None)
print('range',px.index.min(),px.index.max(),'assets',len(frames))
# save signal artifact for audit
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20280717_momentum60_skip5_signal.csv')
