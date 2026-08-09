"""One idea: residual upside peer-dependence change, 20 versus 60 sessions."""
import pandas as pd,numpy as np,json,io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
 src=open('scripts/miner_2_20280601_peer_downside_rebound_participation_residual_40obs.py').read().replace("END=pd.Timestamp('2028-05-31')","END=pd.Timestamp('2028-07-26')")
 exec(compile(src,'library.py','exec'),globals())
p,r,A,END,lib=globals()['p'],globals()['r'],globals()['A'],globals()['END'],globals()['lib']
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
# A fresh upside-state analogue: changes in co-movement when the rest of the universe rises.
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 x=r[a].where(peer[a]>0); y=peer[a].where(peer[a]>0)
 raw[a]=x.rolling(20,min_periods=9).corr(y)-x.rolling(60,min_periods=20).corr(y)
# Reconstruct two later Miner-2 signals and neutralize only direct dependence-family overlaps.
bottom=(r.rank(axis=1,pct=True)<=.2).astype(float);tail=bottom.rolling(60,min_periods=20).mean()
down=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 x=r[a].where(peer[a]<0);y=peer[a].where(peer[a]<0)
 down[a]=x.rolling(20,min_periods=9).corr(y)-x.rolling(60,min_periods=20).corr(y)
controls={'inverse_upside_peer_correlation':lib['inverse_upside_peer_correlation'],'downside_peer_correlation':lib['downside_peer_correlation'],'asymmetric_peer_beta_resilience':lib['asymmetric_peer_beta_resilience'],'peer_downside_tail_persistence_residual':tail,'residual_downside_peer_dependence_change':down}
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y')]+[v.loc[d].rename(k) for k,v in controls.items()],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:].values];f.loc[d,z.index]=z.y.values-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
lib['peer_downside_tail_persistence_residual']=tail;lib['residual_downside_peer_dependence_change']=down
def calc(h):
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);x.index=pd.to_datetime(x.index);sd=x.std();regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std()),'hit_ratio':None if len(q)==0 else float((q>0).mean())}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turns)),'regimes':regs}
print('CANDIDATE residual_upside_peer_dependence_change_20_60 visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date());print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):raise RuntimeError('Missing library correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
