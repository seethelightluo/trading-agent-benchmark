import pandas as pd, numpy as np
x=pd.read_csv('../persistent/factor_signals_miner_2_20270225_intraday_reversal3.csv',parse_dates=['date'])
x=x.dropna(subset=['signal','forward_return'])
def block(lo,hi):
 z=[]; ns=[]
 for dt,g in x[(x.date>=lo)&(x.date<=hi)].groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.forward_return.nunique()>1:
   z.append(g.signal.rank().corr(g.forward_return.rank()));ns.append(len(g))
 z=pd.Series(z)
 print(lo,hi,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2026-07-16','2027-02-24'),('2027-01-01','2027-02-24')]:block(a,b)
q=pd.read_csv('../persistent/factor_signals_miner_3_20270225_dispersion_reversal.csv',parse_dates=['date']).rename(columns={'asset':'symbol','signal':'old'})
a=x[['date','symbol','signal']].copy();a['r']=a.groupby('date').signal.rank(pct=True);q['qr']=q.groupby('date').old.rank(pct=True)
m=a.merge(q,on=['date','symbol']).dropna().groupby('date').apply(lambda g:g.r.corr(g.qr)).dropna()
print('overlap dates',len(m),'mean rank corr',round(m.mean(),4),'median',round(m.median(),4),'absmean',round(m.abs().mean(),4))
print('candidate rows',len(x),'unique dates',x.date.nunique(),'symbols',x.symbol.nunique())
