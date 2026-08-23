import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.close.astype(float)
all_dates=sorted(set().union(*[set(x.index) for x in px.values()]))
rows=[]
for s,x in px.items():
 r=x.pct_change(); eff=(x-x.shift(10)).abs()/r.abs().rolling(10).sum()
 sig=-(np.sign(x-x.shift(10))*eff).shift(1)
 for dt in all_dates:
  if dt in sig.index and pd.notna(sig.loc[dt]): rows.append((dt,s,float(sig.loc[dt])))
f=pd.DataFrame(rows,columns=['date','symbol','factor']).set_index(['date','symbol']).sort_index()
for h in [1,3,5,10]:
 rr=[]
 for s,x in px.items():
  fr=(x.shift(-h)/x-1).rename('fwd'); rr.append(fr.to_frame().assign(symbol=s).reset_index().set_index(['date','symbol']))
 z=f.join(pd.concat(rr),how='inner').dropna()
 ics=z.groupby(level=0).apply(lambda q:q.factor.corr(q.fwd) if len(q)>=8 else np.nan).dropna()
 print('H',h,'IC',round(ics.mean(),6),'ICIR',round(ics.mean()/ics.std(ddof=1),6),'hit',round((ics>0).mean(),4),'dates',len(ics),'avgN',round(z.groupby(level=0).size().mean(),2),'coverage',round(len(z)/sum(len(x) for x in px.values()),4))
 for label,lo,hi in [('early','2026-07-20','2027-12-31'),('late','2028-01-01','2029-04-05'),('recent','2028-10-01','2029-04-05')]:
  a=ics[(ics.index>=lo)&(ics.index<=hi)]; print(label,round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),len(a))
print('assets',len(px),'raw dates',len(all_dates))
