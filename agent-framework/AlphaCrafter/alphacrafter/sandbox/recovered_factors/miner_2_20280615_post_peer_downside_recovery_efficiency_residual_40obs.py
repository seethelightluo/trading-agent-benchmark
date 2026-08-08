"""One idea: 40-session post-peer-downside recovery efficiency residual."""
import pandas as pd,numpy as np,json
# Exact, full admitted-library reconstruction from prior script; candidate only differs.
src=open('scripts/miner_2_20280601_peer_downside_rebound_participation_residual_40obs.py').read()
src=src.replace("END=pd.Timestamp('2028-05-31')","END=pd.Timestamp('2028-06-14')")
exec(compile(src,'library.py','exec'),globals())
p,r,A,END,lib=globals()['p'],globals()['r'],globals()['A'],globals()['END'],globals()['lib']
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
bottom=(r.rank(axis=1,pct=True)<=.2).astype(float);tail=bottom.rolling(60,min_periods=20).mean()
# On days completing a five-session path begun immediately after a broad peer decline,
# measure asset's cumulative excess return vs peers; average eligible paths over 40 sessions.
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 path=r[a].rolling(5).sum()-peer[a].rolling(5).sum()
 raw[a]=path.where(peer[a].shift(5)<0).rolling(40,min_periods=10).mean()
# Remove ordinary trend and persistence to isolate conditional recovery-path behavior.
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('trend'),tail.loc[d].rename('tail')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['trend','tail']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);x.index=pd.to_datetime(x.index);sd=x.std();regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std()),'hit_ratio':None if len(q)==0 else float((q>0).mean())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE post_peer_downside_recovery_efficiency_residual_40obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):raise RuntimeError('Missing library correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
