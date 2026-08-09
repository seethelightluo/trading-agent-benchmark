"""Revalidation only: Asymmetric Peer Beta Resilience (40 observations), visible through 2028-04-19."""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-04-19')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def cbeta(x,y,mask):
 return x.where(mask).rolling(40,min_periods=12).cov(y.where(mask))/y.where(mask).rolling(40,min_periods=12).var()
f=pd.DataFrame({a:cbeta(r[a],peer[a],peer[a]<0)-cbeta(r[a],peer[a],peer[a]>0) for a in A})
def calc(h):
 fw=p.shift(-h)/p-1; obs=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('fw')],axis=1).dropna()
  if len(z)>=8: obs.append((d,z.f.corr(z.fw,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(obs)); sd=x.std()
 regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(ys)]; regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_standard_error':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'mean_rank_turnover_10obs':float(np.mean(turn)),'regimes':regs}
print('REVALIDATION asymmetric_peer_beta_resilience_40obs visible',END.date(),'range',p.index.min().date(),p.index.max().date(),'assets',len(A))
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]: print('METRIC',json.dumps(calc(h)))
# Last 120 valid IC dates highlights current drift independently of full-history admission result.
fw=p.shift(-10)/p-1; x=[]
for d in f.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:x.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.Series(dict(x)); q=x.tail(120); print('RECENT120',json.dumps({'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std()),'hit_ratio':float((q>0).mean())}))
