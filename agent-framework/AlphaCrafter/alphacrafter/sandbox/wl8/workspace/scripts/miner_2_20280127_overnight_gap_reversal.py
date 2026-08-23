import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-01-26')
O={}; C={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=END].set_index('date').sort_index(); O[s]=x.open; C[s]=x.close
op=pd.DataFrame(O).sort_index(); cl=pd.DataFrame(C).sort_index()
# Signal available after close t: reverse the overnight gap observed at t, predict t+1 close-to-close return.
gap=op/cl.shift(1)-1
sig=-gap
fwd=cl.shift(-1)/cl-1
vals=[]; ns=[]; dates=[]
for dt in cl.index:
 g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q); ns.append(len(g)); dates.append(dt)
a=np.array(vals)
print('factor overnight_gap_reversal dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,mask in [('2020-22',pd.Series(cl.index.year<=2022,index=cl.index)),('2023-25',pd.Series((cl.index.year>=2023)&(cl.index.year<=2025),index=cl.index)),('2026',pd.Series(cl.index.year==2026,index=cl.index)),('2027+',pd.Series(cl.index.year>=2027,index=cl.index)),('recent180',pd.Series(cl.index>=END-pd.Timedelta(days=180),index=cl.index))]:
 z=a[[i for i,d in enumerate(dates) if mask.loc[d]]]
 print(label,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
# non-overlapping 5d decay for context, not admission
for h in [2,5,10]:
 ff=cl.shift(-h)/cl-1; vv=[]
 for dt in cl.index:
  g=pd.DataFrame({'s':sig.loc[dt],'f':ff.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vv.append(spearmanr(g.s,g.f).statistic)
 print('horizon',h,'IC',round(np.nanmean(vv),6),'n',len(vv))
