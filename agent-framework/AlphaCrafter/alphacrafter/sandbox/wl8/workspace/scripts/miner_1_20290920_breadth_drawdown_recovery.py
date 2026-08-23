import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
# Candidate: drawdown recovery, activated only when lagged cross-asset breadth is weak (<= median); lagged and volatility normalized.
allr=[]
for s,x in D.items():
 p=x.close.astype(float); allr.append(p.pct_change(20).rename(s))
R=pd.concat(allr,axis=1); breadth=(R>0).mean(axis=1).shift(1); threshold=breadth.rolling(252,min_periods=100).median().shift(1)
rows=[]
for s,x in D.items():
 p=x.close.astype(float); r=p.pct_change(); trough=p.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std()
 base=np.log(p/trough)/vol.replace(0,np.nan)
 f=base.where(breadth<=threshold).shift(1); fw=np.log(p.shift(-10)/p.shift(-1))
 rows.append(pd.DataFrame({'date':p.index,'symbol':s,'factor':f,'fwd':fw}))
a=pd.concat(rows,ignore_index=True).dropna(); a=a[a.date<=pd.Timestamp('2029-09-18')]
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ics.append(g.factor.corr(g.fwd,method='spearman'))
ic=pd.Series(ics).dropna(); print('dates',len(ic),'avg_instruments',round(a.groupby('date').size().loc[a.groupby('date').size()>=8].mean(),2),'coverage',round(len(a)/(len(a.date.unique())*15),4)); print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4))
for h in [1,5,10,20]:
 q=[]
 for s,x in D.items():
  p=x.close.astype(float); r=p.pct_change(); tr=p.rolling(60,min_periods=40).min(); v=r.rolling(20,min_periods=15).std(); f=(np.log(p/tr)/v.replace(0,np.nan)).where(breadth<=threshold).shift(1); fw=np.log(p.shift(-h)/p.shift(-1)); q.append(pd.DataFrame({'date':p.index,'f':f,'y':fw}))
 z=pd.concat(q,ignore_index=True).dropna(); vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append(g.f.corr(g.y,method='spearman'))
 ss=pd.Series(vals).dropna(); print('h',h,'IC',round(ss.mean(),6),'ICIR',round(ss.mean()/ss.std(),6),'n',len(ss))
print('regimes')
# recompute dated IC
icd=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: icd.append((dt,g.factor.corr(g.fwd,method='spearman')))
s=pd.Series(dict(icd));
for n,m in [('2026',s.index.year==2026),('2027-28',s.index.year.isin([2027,2028])),('recent360',s.index>=s.index.max()-pd.Timedelta(days=360)),('recent180',s.index>=s.index.max()-pd.Timedelta(days=180))]:
 q=s[m].dropna(); print(n,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if q.std()>0 else None)
print('turnover',round(float(a.pivot(index='date',columns='symbol',values='factor').rank(pct=True).diff().abs().mean().mean()),6))
