"""One idea: 40-observation inverse idiosyncratic volatility residual."""
import pandas as pd,numpy as np,json
src=open('scripts/miner_3_20280309_residual_trend_path_efficiency_40obs.py').read().replace("END=pd.Timestamp('2028-03-08')","END=pd.Timestamp('2028-06-14')")
exec(compile(src,'library.py','exec'),globals())
p,r,A,END,lib=globals()['p'],globals()['r'],globals()['A'],globals()['END'],globals()['lib']
# Defensive quality: low residual volatility after removing the asset's 40d peer beta.
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
b=pd.DataFrame({a:r[a].rolling(40,min_periods=30).cov(peer[a])/peer[a].rolling(40,min_periods=30).var() for a in A})
res=pd.DataFrame({a:r[a]-b[a]*peer[a] for a in A})
raw=-res.rolling(40,min_periods=30).std()
# Residualize known total-tail-risk signals cross-sectionally, retaining only idiosyncratic component.
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/r[a].rolling(20,min_periods=15).std() for a in A})
kurt=-r.rolling(40,min_periods=30).kurt()
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),es.loc[d].rename('es'),kurt.loc[d].rename('k')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['es','k']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# Add post-March admitted peer-tail signal to the complete library screen.
csbot=pd.DataFrame(index=r.index,columns=A,dtype=float)
for d in r.index:
 q=r.loc[d].quantile(.2);csbot.loc[d]=(r.loc[d]<=q).astype(float)
tail=csbot.rolling(60,min_periods=45).mean(); trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std();pt=tail*np.nan
for d in p.index:
 z=pd.concat([tail.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];pt.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['peer_downside_tail_persistence_residual']=pt
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);x.index=pd.to_datetime(x.index);sd=x.std();regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('CANDIDATE inverse_idiosyncratic_volatility_residual_40obs visible',END.date(),'assets',len(A),'range',p.index.min().date(),p.index.max().date())
print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho):raise RuntimeError('Missing library correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
