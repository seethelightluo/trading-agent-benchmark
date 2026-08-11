import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=d[d.close.notna()]
# downside asymmetry: positive-return participation minus downside participation, normalized by total volatility
# At date t use the preceding 20 completed returns only.
for window in [10,20,40]:
 for h in [5,10,20]:
  rows=[]
  for s,x in D.items():
   ix=x.index
   for j in range(window, len(ix)-h):
    rr=x.close.pct_change().iloc[j-window+1:j+1].to_numpy()
    y=x.close.iloc[j+h]/x.close.iloc[j]-1
    if len(rr)==window and np.isfinite(rr).all() and np.isfinite(y):
     pos=rr[rr>0]; neg=rr[rr<0]
     if len(pos)>=2 and len(neg)>=2:
      # high upside frequency/magnitude relative to downside is favorable
      f=(pos.mean()+1e-8)/(abs(neg.mean())+1e-8) * (len(pos)/window)
      rows.append((ix[j],s,f,y))
  a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]; ds=[]; ns=[]
  for dt,g in a.groupby('date'):
   g=g.dropna()
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    q.append(spearmanr(g.f,g.y).statistic); ds.append(dt); ns.append(len(g))
  z=np.array(q)
  print('window',window,'horizon',h,'dates',len(z),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
  if h==10 and len(z): print('annual',{y:round(z[np.array([d.year for d in ds])==y].mean(),6) for y in sorted(set(d.year for d in ds))})
# rank turnover for selected 20d signal
prev=None; ts=[]; valid=0
for dt in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={}
 for s,x in D.items():
  if dt not in x.index: continue
  j=x.index.get_loc(dt)
  if j<20: continue
  rr=x.close.pct_change().iloc[j-19:j+1].to_numpy(); pos=rr[rr>0]; neg=rr[rr<0]
  if len(pos)>=2 and len(neg)>=2: vals[s]=(pos.mean()+1e-8)/(abs(neg.mean())+1e-8)*(len(pos)/20)
 z=pd.Series(vals).dropna()
 if len(z)>=8:
  valid+=1; rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index); ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk
print('turnover',round(np.mean(ts),6),'turnover_dates',valid)
