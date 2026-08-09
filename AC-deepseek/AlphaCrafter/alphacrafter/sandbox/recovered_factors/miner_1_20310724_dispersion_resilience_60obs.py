import pandas as pd,numpy as np,json,glob,os
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2031-07-23')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change(); med=r.median(axis=1); disp=r.std(axis=1)
# Candidate: rolling relative resilience to high-dispersion days, normalized by dispersion variance.
rel=r.sub(med,axis=0); z=(disp-disp.rolling(60,min_periods=35).mean())/disp.rolling(60,min_periods=35).std()
f=pd.DataFrame({a:rel[a].rolling(60,min_periods=35).cov(z)/z.rolling(60,min_periods=35).var() for a in A})
# cross-sectional residualize against broad admitted-style families to make signal interpretable/orthogonal
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  q=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(q)>=8:
   X=np.c_[np.ones(len(q)),q.iloc[:,1:]]; o.loc[d,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return o
# broad library reconstruction (definitions of all admitted current factors)
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mask=None):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(w,min_periods=12).cov(y)/y.rolling(w,min_periods=12).var().replace(0,np.nan)
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt()
vup=pd.DataFrame({a:r[a].where(z>0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
vdown=pd.DataFrame({a:r[a].where(z<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
dxy=rd('000300.SH')*np.nan # no macro via stock API; proxy omitted
lib={'risk_trend':trend,'reversal':rev,'down':down,'es':es,'kurt':kurt,'vup':vup,'vdown':vdown,'peer_beta':pd.DataFrame({a:beta(r[a],peer[a]) for a in A})}
f=resid(f,[trend,rev,down,es,kurt,vup,vdown])
def met(h):
 fw=p.shift(-h)/p-1; q=[]; ns=[]
 for d in f.index:
  x=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(x)>=8:q.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ns.append(len(x))
 s=pd.Series(q); return dict(horizon=h,dates=len(s),mean_n=np.mean(ns),ic=s.mean(),icir=s.mean()/s.std(),hit=(s>0).mean())
print('VISIBLE',p.index.max().date(),'dates',len(p),'assets',len(A),'coverage',f.count().sum()/f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(met(h)))
mx=0; who=''
for n,x in lib.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna(); rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 print('LIB',n,round(rho,6),len(q))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
