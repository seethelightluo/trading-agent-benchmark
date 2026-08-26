import numpy as np, pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-10-17'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
# Beta-neutral short-horizon reversal: remove each asset's rolling 60d beta to the
# equal-weight cross-asset market return, then fade the idiosyncratic 5d shock.
P=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index(); R=P.pct_change()
M=R.mean(axis=1); rows=[]
for s,x in D.items():
 c=x.close.astype(float); r=R[s].reindex(c.index); mm=M.reindex(c.index)
 cov=r.rolling(60,min_periods=40).cov(mm); vm=mm.rolling(60,min_periods=40).var(); beta=cov/vm.replace(0,np.nan)
 resid5=r.rolling(5,min_periods=5).sum()-beta*mm.rolling(5,min_periods=5).sum()
 vol=r.rolling(30,min_periods=20).std(); f=-resid5/vol.replace(0,np.nan)
 y=c.shift(-10)/c-1
 rows.append(pd.DataFrame({'date':c.index,'symbol':s,'factor':f,'fwd':y}))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
ics=[]
for d,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3: ics.append((d,g.factor.corr(g.fwd),len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n']).sort_values('date')
print('cutoff',cut.date(),'rows',len(a),'dates',len(ic),'avgN',round(ic.n.mean(),2),'coverage',round(len(a)/(len(D)*len(P.index)),4))
print('IC',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(ddof=1),'hit',(ic.ic>0).mean(),'median',ic.ic.median())
for h in [1,5,10,20]:
 vals=[]
 for s,x in D.items():
  c=x.close.astype(float); r=R[s].reindex(c.index); mm=M.reindex(c.index); beta=r.rolling(60,min_periods=40).cov(mm)/mm.rolling(60,min_periods=40).var(); f=-(r.rolling(5).sum()-beta*mm.rolling(5).sum())/r.rolling(30,min_periods=20).std(); z=pd.DataFrame({'date':c.index,'f':f,'y':c.shift(-h)/c-1}).dropna(); vals.append(z)
 q=pd.concat(vals).reset_index(drop=True); out=[]
 for d,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>2: out.append(g.f.corr(g.y))
 v=pd.Series(out).dropna(); print('H',h,'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'dates',len(v))
os.makedirs('scripts',exist_ok=True); a[['date','symbol','factor']].to_csv('scripts/miner_1_20321018_beta_neutral_reversal_signal.csv',index=False)
