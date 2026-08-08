"""One candidate: orthogonal joint dollar-and-volatility-relief resilience."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-11-15')
def read(a,index=False):
 d=(get_index_daily_data(a,5000) if index else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:read(a) for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=12,cond=None):
 if cond is not None:x=x.where(cond);y=y.where(cond)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
vix=read('VIX',True).pct_change().reindex(r.index);dxy=read('DXY',True).pct_change().reindex(r.index);cny=read('USDCNY',True).pct_change().reindex(r.index)
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
upraw=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A});vdown=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
dup=pd.DataFrame({a:r[a].where(dxy>0).rolling(40,min_periods=12).mean()/vol[a] for a in A});ddown=pd.DataFrame({a:r[a].where(dxy<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
# Candidate: relative return per unit volatility on joint dollar and volatility relief days, stripped of individual relief effects and broad defensive controls.
joint=pd.DataFrame({a:r[a].where((vix<0)&(dxy<0)).rolling(60,min_periods=12).mean()/vol[a] for a in A})
f=resid(joint,[es,down,kurt,trend,upraw,vdown,dup,ddown])
def metric(h):
 fw=p.shift(-h)/p-1;xs=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:xs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs));sd=x.std(); regs={}
 for n,s in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[s];regs[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std()}
 ts=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(ts)),'regimes':regs}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'candidate_cells',int(f.count().sum()),'of',f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h),default=float))
# Reconstruct every active signal with available price/macro evidence.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=resid(acc,[trend]);spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dxy,40,12,dxy<0) for a in A});persist=r.le(r.quantile(.2,axis=1),axis=0).astype(float).rolling(60,min_periods=40).mean();dep=resid(pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A})-down,[down]);vp=resid(pd.DataFrame({a:r[a].where(vix>vix.rolling(60,min_periods=40).quantile(.7)).rolling(60,min_periods=15).corr(peer[a].where(vix>vix.rolling(60,min_periods=40).quantile(.7))) for a in A}),[es,down,kurt,trend]);cr=resid(pd.DataFrame({a:-beta(r[a],cny,60,15,cny>cny.rolling(60,min_periods=40).quantile(.7)) for a in A}),[du,es,down,trend]);vup=resid(upraw,[es,down,kurt,trend]);vdr=resid(vdown,[es,down,kurt,trend,upraw])
lib={'ravmom':trend,'reversal':rev,'downside_corr':down,'autocorr':aut,'vix_stress':vp,'cny_stress':cr,'asym_beta':asym,'down_dependence_change':dep,'trend_accel':orth,'negative_spx_beta':spx,'inverse_kurtosis':kurt,'inverse_es':es,'inverse_up_corr':upinv,'negative_dxy_up':du,'positive_dxy_down':dd,'vix_up_resilience':vup,'vix_down_resilience':vdr}
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan;print('LIB',n,'rho',rho,'cells',len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
