"""One idea: VIX-up conditional correlation independence, residualized against unconditional independence."""
import runpy,pandas as pd,numpy as np,json
src=open('scripts/miner_3_20280309_residual_trend_path_efficiency_40obs.py').read().replace("END=pd.Timestamp('2028-03-08')","END=pd.Timestamp('2028-04-05')")
exec(compile(src,'library.py','exec'),globals())
p,r,A,END,lib=globals()['p'],globals()['r'],globals()['A'],globals()['END'],globals()['lib']
# Add the factor admitted in the last cycle to the mandatory library comparison.
base=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 q=r.iloc[t-39:t+1]; c=q.corr(min_periods=30)
 for a in A:
  v=c.loc[a].drop(a).abs().dropna()
  if len(v)>=8: base.loc[p.index[t],a]=-v.mean()
lib['inverse_cross_asset_correlation_concentration']=base
# Conditional signal: independent return paths specifically across VIX-rising days. Remove cross-sectional unconditional-independence exposure.
vi=get_index_daily_data('VIX',5000).set_index('date');vi.index=pd.to_datetime(vi.index)
vret=pd.to_numeric(vi.loc[:END,'close'],errors='coerce').pct_change().reindex(p.index)
raw=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 q=r.iloc[t-39:t+1]; mask=vret.iloc[t-39:t+1]>0; c=q.where(mask,axis=0).corr(min_periods=12)
 for a in A:
  v=c.loc[a].drop(a).abs().dropna()
  if len(v)>=8: raw.loc[p.index[t],a]=-v.mean()
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('raw'),base.loc[d].rename('base')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.base]; f.loc[d,z.index]=z.raw-X@np.linalg.lstsq(X,z.raw,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(); regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  z=x[x.index.year.isin(ys)];regs[n]={'dates':len(z),'ic':None if len(z)==0 else float(z.mean()),'icir':None if len(z)<2 or z.std()==0 else float(z.mean()/z.std())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE vix_up_residual_correlation_independence_40obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman'); print('LIB',n,rho,len(z))
 if not np.isfinite(rho): raise RuntimeError('Missing library correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
