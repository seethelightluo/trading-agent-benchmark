import numpy as np, pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-09-19')
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
# Candidate: range-normalized short-term reversal, with a slow trend anchor.
# factor is contrarian 3d shock divided by median true range over 20d, attenuated by 60d trend.
rows=[]
for s,x in D.items():
    c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float)
    tr=(h-l)/c.replace(0,np.nan)
    r3=c.pct_change(3); r60=c.pct_change(60)
    scale=tr.rolling(20,min_periods=15).median()
    f=(-r3/scale)*(1+0.35*np.tanh(-r60/0.15))
    fr=c.shift(-10)/c-1
    z=pd.DataFrame({'date':c.index,'symbol':s,'factor':f,'fwd':fr,'r3':r3,'scale':scale}).dropna()
    rows.append(z)
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=['factor','fwd'])
ics=[]
for d,g in a.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3:
        ics.append((d,g.factor.corr(g.fwd),len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n'])
print('cutoff',cut.date(),'rows',len(a),'dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/sum(max(0,len(x)-70) for x in D.values()))
print('IC',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(ddof=1),'hit',(ic.ic>0).mean(),'med',ic.ic.median())
for k in [0, len(ic)//3, 2*len(ic)//3]:
 q=ic.iloc[k:(len(ic) if k==2*len(ic)//3 else k+len(ic)//3)]; print('third',k,q.ic.mean(),len(q))
# signal artifact for provenance
os.makedirs('scripts',exist_ok=True); a[['date','symbol','factor']].to_csv('scripts/miner_3_20320920_range_shock_reversal_signal.csv',index=False)
for h in [1,5,10,20]:
 # recompute horizon corr with same factor
 rr=[]
 for s,x in D.items():
  c=x.close.astype(float); tr=(x.high-x.low)/c; r3=c.pct_change(3); r60=c.pct_change(60); scale=tr.rolling(20,min_periods=15).median(); f=(-r3/scale)*(1+0.35*np.tanh(-r60/0.15))
  z=pd.DataFrame({'date':c.index,'f':f,'y':c.shift(-h)/c-1}).dropna(); rr.append(z)
 q=pd.concat(rr).reset_index(drop=True)
 vals=[]
 for d,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>2: vals.append(g.f.corr(g.y))
 vals=pd.Series(vals).dropna(); print('H',h,'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1),'dates',len(vals))
