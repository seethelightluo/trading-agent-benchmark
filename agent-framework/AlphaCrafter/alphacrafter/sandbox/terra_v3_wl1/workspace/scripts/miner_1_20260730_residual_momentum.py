import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None: D[s]=d.sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().pct_change(); rows=[]
for i in range(70,len(P)-1):
 x=P.iloc[i-60:i]['SPX'];f={}
 for s in U:
  y=P[s].iloc[i-60:i];ok=x.notna()&y.notna()
  if ok.sum()<20: continue
  beta=np.cov(y[ok],x[ok],ddof=1)[0,1]/max(np.var(x[ok],ddof=1),1e-12)
  a=P[s].iloc[i-20:i];b=P['SPX'].iloc[i-20:i];ok2=a.notna()&b.notna()
  if ok2.sum()>=10:f[s]=float(a[ok2].sum()-beta*b[ok2].sum())
 fu=P.iloc[i+1];ok=set(f)&set(fu.dropna().index)
 if len(ok)>=8:rows.append((f,{s:fu[s] for s in ok}))
ics=[]; prev=None; turns=[]; ni=[]
for f,fu in rows:
 a=pd.Series({s:f[s] for s in fu});b=pd.Series(fu);ic=a.corr(b,method='spearman');ics.append(ic);ni.append(len(a));r=a.rank(pct=True)
 if prev is not None: turns.append(np.mean(r.sub(prev,fill_value=.5).abs()))
 prev=r
z=np.array(ics);print('dates',len(z),'avg_names',np.mean(ni),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for a,b in [(0,700),(700,1200),(1200,9999)]:
 q=z[a:b];print('regime',a,b,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
for h in [5,10]:
 q=[]
 for i in range(70,len(P)-h):
  x=P.iloc[i-60:i]['SPX'];f={}
  for s in U:
   y=P[s].iloc[i-60:i];ok=x.notna()&y.notna()
   if ok.sum()<20:continue
   beta=np.cov(y[ok],x[ok],ddof=1)[0,1]/max(np.var(x[ok],ddof=1),1e-12);a=P[s].iloc[i-20:i];b=P['SPX'].iloc[i-20:i];ok2=a.notna()&b.notna()
   if ok2.sum()>=10:f[s]=a[ok2].sum()-beta*b[ok2].sum()
  fu=P.iloc[i+1:i+1+h].sum();ok=set(f)&set(fu.dropna().index)
  if len(ok)>=8:q.append(pd.Series({s:f[s] for s in ok}).corr(pd.Series({s:fu[s] for s in ok}),method='spearman'))
 q=np.array(q);print('h',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
