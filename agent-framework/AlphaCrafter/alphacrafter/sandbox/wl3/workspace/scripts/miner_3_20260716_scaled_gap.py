import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index() for s in U}
# Candidate: volatility-scaled overnight gap continuation, distinct from CLV and multi-day reversal
rows=[]
for s,x in D.items():
    ret=x.close.pct_change(); gap=x.open/x.close.shift(1)-1
    tr=(x.high-x.low)/x.close.shift(1)
    vol=ret.rolling(20,min_periods=15).std()
    f=(gap/vol).replace([np.inf,-np.inf],np.nan)
    # forward next valid observation return
    fw=x.close.shift(-1)/x.close-1
    z=pd.DataFrame({'date':x.index,'f':f,'fw':fw,'s':s,'r5':x.close/x.close.shift(5)-1,'clv':-(2*(x.close-x.low)/(x.high-x.low)-1),'mom':x.close/x.close.shift(20)-1})
    rows.append(z)
a=pd.concat(rows,ignore_index=True).dropna(subset=['f','fw'])
ics=[]; turns=[]; counts=[]
for d,g in a.groupby('date'):
    if len(g)>=8:
      ic=g.f.corr(g.fw,method='spearman'); ics.append(ic); counts.append(len(g));
      turns.append(g.f.rank(pct=True).sub(.5).abs().mean())
ics=np.array(ics)
print('candidate scaled_gap_continuation dates',len(ics),'meanN',np.mean(counts),'coverage',len(a)/sum(len(x) for x in D.values()))
print('IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0),'std',np.nanstd(ics,ddof=1),'turn_proxy',np.mean(turns))
for h in [5,10]:
 vals=[]
 for s,x in D.items():
  f=(x.open/x.close.shift(1)-1)/x.close.pct_change().rolling(20,min_periods=15).std()
  fw=x.close.shift(-h)/x.close-1
  vals.append(pd.DataFrame({'date':x.index,'f':f,'fw':fw}))
 q=pd.concat(vals,ignore_index=True).dropna()
 z=[g.f.corr(g.fw,method='spearman') for d,g in q.groupby('date') if len(g)>=8]
 print(str(h)+'d',len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
# overlap date demean pooled
for c in ['r5','clv','mom']:
 q=a[['date','f',c]].dropna(); q['x']=q.f-q.groupby('date').f.transform('mean');q['y']=q[c]-q.groupby('date')[c].transform('mean');print('corr',c,q.x.corr(q.y))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=ics[[str(d).startswith(tuple([str(y) for y in range(int(lo),int(hi)+1)])) for d in sorted(set(a.date))]] if False else []
# yearly
for y,g in a.groupby(a.date.dt.year):
 z=[h.f.corr(h.fw,method='spearman') for d,h in g.groupby('date') if len(h)>=8]
 print('year',y,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1) if len(z)>1 else np.nan)
