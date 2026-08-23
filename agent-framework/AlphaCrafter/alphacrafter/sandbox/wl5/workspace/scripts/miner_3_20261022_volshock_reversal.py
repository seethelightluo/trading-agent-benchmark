import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.drop_duplicates('date').set_index('date').sort_index()
rows=[]; total=0
for s,d in D.items():
 r=d.close.pct_change(); total+=len(d)
 # contrarian return, emphasized in volatility expansion but bounded for robustness
 sv=r.rolling(5,min_periods=5).std(); lv=r.rolling(30,min_periods=20).std()
 shock=(sv/lv).clip(0.25,4.0)
 sig=-r.rolling(3,min_periods=3).sum()*shock
 fr=d.close.shift(-1)/d.close-1
 rows.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr,'s':s}).dropna())
x=pd.concat(rows,ignore_index=True); ics=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
  v=g.sig.corr(g.fr,method='spearman')
  if np.isfinite(v): ics.append(v);ns.append(len(g))
a=np.array(ics); print('dates',len(a),'avg_n',np.mean(ns),'assets',x.s.nunique(),'coverage',len(x)/total)
print('1d IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for h in [5,10]:
 q=[]
 for s,d in D.items():
  r=d.close.pct_change(); shock=(r.rolling(5,min_periods=5).std()/r.rolling(30,min_periods=20).std()).clip(.25,4)
  sig=-r.rolling(3,min_periods=3).sum()*shock; fr=d.close.shift(-h)/d.close-1
  q.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr,'s':s}).dropna())
 q=pd.concat(q,ignore_index=True); z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
   v=g.sig.corr(g.fr,method='spearman')
   if np.isfinite(v):z.append(v)
 z=np.array(z);print(str(h)+'d dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
x.to_csv('scripts/miner_3_20261022_volshock_reversal_signal.csv',index=False)
