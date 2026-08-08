"""One idea: residual relative downside-semivolatility compression (60d)."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-05-17')
def rd(a,ix=False):
 d=(get_index_daily_data(a,5000) if ix else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=12,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
vix=rd('VIX',True).pct_change().reindex(r.index);dxy=rd('DXY',True).pct_change().reindex(r.index)
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A});vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A});du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dxy,40,12,dxy<0) for a in A})
# Higher score = downside deviations have compressed relative to upside deviations over 60 observations.
dsemi=(-r.clip(upper=0)).pow(2).rolling(60,min_periods=40).mean().pow(.5)
usemi=r.clip(lower=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
raw=-np.log((dsemi+1e-10)/(usemi+1e-10))
# Isolate asymmetry from broad tail-risk, dependence, distribution shape and trend exposures.
f=resid(raw,[es,down,kurt,trend,vu,vd,du,dd])
def metrics(h):
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 reg={}
 for n,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029YTD',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[m];reg[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std()}
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/x.std(),'hit_ratio':(x>0).mean(),'dates':len(x),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':reg}
print('VISIBLE',p.index.max().date(),'assets',len(A),'price_dates',len(p),'factor_cells',int(f.count().sum()),'possible_cells',f.size,'coverage',float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h)))
# Complete reconstructed library evidence, including active and retained historical signals.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=resid((p/p.shift(20)-p.shift(20)/p.shift(60))/vol,[trend]);spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A});isk=resid(-idres.rolling(40,min_periods=30).skew(),[es,down,kurt,trend]);vdown=resid(vd,[es,down,kurt,trend,vu]);vup=resid(vu,[es,down,kurt,trend]);persist=r.le(r.quantile(.2,axis=1),axis=0).astype(float).rolling(60,min_periods=40).mean();tail=resid(persist,[trend]);shortdep=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});dep=resid(shortdep-down,[down]);prior=-resid(pd.DataFrame({a:r[a].where(vix<0).rolling(80,min_periods=24).mean()/vol[a] for a in A})-pd.DataFrame({a:r[a].where(vix<0).rolling(20,min_periods=8).mean()/vol[a] for a in A}),[vd,vu,es,down,kurt,trend,du,dd])
lib={'ravmom':trend,'reversal':rev,'downside_peer_correlation':down,'return_autocorrelation':aut,'trend_acceleration':acc,'negative_spx_beta':spx,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'inverse_upside_correlation':up,'negative_dxy_up_beta':du,'positive_dxy_down_beta':dd,'inverse_idiosyncratic_skew':isk,'vix_up_resilience':vup,'vix_down_resilience':vdown,'asymmetric_peer_resilience':asym,'tail_persistence':tail,'downside_dependence_change':dep,'vix_relief_fragility':prior}
mx=0;who=''
for n,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');print('LIB',n,round(rho,6),len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,who)
