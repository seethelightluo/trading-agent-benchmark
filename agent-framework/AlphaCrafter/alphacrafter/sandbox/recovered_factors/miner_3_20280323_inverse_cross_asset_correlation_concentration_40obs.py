"""One idea: inverse cross-asset correlation concentration, a diversification-resilience signal."""
import runpy,pandas as pd,numpy as np,json
# Reuse the visible-data universe and exact admitted-library reconstruction from prior validation.
z=runpy.run_path('scripts/miner_3_20280309_residual_trend_path_efficiency_40obs.py')
p,r,A,END,lib=z['p'],z['r'],z['A'],z['END'],z['lib']
# At each date, calculate each asset's 40-observation mean absolute correlation to the other 14 assets.
# Negative mean correlation = more independent return path; cross-sectional ranks form a long-only defensive signal.
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for t in range(39,len(p)):
 q=r.iloc[t-39:t+1]
 c=q.corr(min_periods=30)
 for a in A:
  v=c.loc[a].drop(a).abs().replace([np.inf],np.nan).dropna()
  if len(v)>=8:f.loc[p.index[t],a]=-v.mean()
def calc(h):
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8:out.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out)); sd=x.std(); regs={}
 for n,ys in {'2020_21':[2020,2021],'2022_23':[2022,2023],'2024_25':[2024,2025],'2026':[2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(ys)];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std()==0 else float(q.mean()/q.std())}
 turn=[]
 for i in range(10,len(f),10):
  q=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn))),'regimes':regs}
print('CANDIDATE inverse_cross_asset_correlation_concentration_40obs visible',END.date(),'assets',len(A),'coverage',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(calc(h)))
mx=-1;who=None
for n,x in lib.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('LIB',n,rho,len(q))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
