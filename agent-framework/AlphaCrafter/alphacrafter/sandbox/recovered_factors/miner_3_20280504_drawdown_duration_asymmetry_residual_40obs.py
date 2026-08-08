"""One idea: 40-session drawdown-duration asymmetry residualized against trend and correlation concentration."""
import pandas as pd,numpy as np,json
src=open('scripts/miner_3_20280309_residual_trend_path_efficiency_40obs.py').read().replace("END=pd.Timestamp('2028-03-08')","END=pd.Timestamp('2028-05-03')")
exec(compile(src,'library.py','exec'),globals())
p,r,A,END,lib=globals()['p'],globals()['r'],globals()['A'],globals()['END'],globals()['lib']
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
# Current loss from trailing peak, scaled by the age of that peak. High scores denote a rapid, fresh drawdown;
# low scores denote a persistent drawdown. Residualization removes ordinary trend and unconditional peer-independence.
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 w=p.iloc[t-39:t+1]
 for a in A:
  x=w[a].dropna()
  if len(x)>=30:
   peak=x.max(); age=len(x)-1-np.flatnonzero(x.to_numpy()==peak)[-1]
   raw.loc[p.index[t],a]=-(p.loc[p.index[t],a]/peak-1)/(age+1)
base=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 c=r.iloc[t-39:t+1].corr(min_periods=30)
 for a in A:
  z=c.loc[a].drop(a).abs().dropna()
  if len(z)>=8: base.loc[p.index[t],a]=-z.mean()
lib['inverse_cross_asset_correlation_concentration']=base
# cross-sectional residual of the single drawdown-path idea against two distinct admitted exposures
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('trend'),base.loc[d].rename('ind')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['trend','ind']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std();regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE drawdown_duration_asymmetry_residual_40obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):raise RuntimeError('Missing library correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
