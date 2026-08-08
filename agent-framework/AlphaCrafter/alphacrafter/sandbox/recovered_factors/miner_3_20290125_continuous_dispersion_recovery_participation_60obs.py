"""One candidate: continuous dispersion-weighted recovery participation, full-history validation."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-01-24')
def rd(a,ix=False):
 d=(get_index_daily_data(a,5000) if ix else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A});r=p.pct_change();v=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=12,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def residual(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
trend=(p/p.shift(20)-1)/v
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
vix=rd('VIX',True).pct_change().reindex(r.index);dxy=rd('DXY',True).pct_change().reindex(r.index);cny=rd('USDCNY',True).pct_change().reindex(r.index)
vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/v[a] for a in A});vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/v[a] for a in A});du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dxy,40,12,dxy<0) for a in A})
# Each recovery return is weighted continuously by trailing standardized cross-asset dispersion, clipped to positive values; avoids rare event filter.
disp=r.std(axis=1); dz=(disp-disp.rolling(60,min_periods=40).mean())/disp.rolling(60,min_periods=40).std(); w=dz.clip(0,3).where(peer.mean(axis=1)>0,0)
raw=pd.DataFrame({a:(r[a]*w).rolling(60,min_periods=30).mean()/v[a] for a in A})
f=residual(raw,[es,down,kurt,trend,vu,vd,du,dd])
def met(h):
 fw=p.shift(-h)/p-1;q=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q)); sd=x.std();regs={}
 for n,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[m];regs[n]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std() if y.std() else np.nan}
 t=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'se':sd/np.sqrt(len(x)),'n':np.mean(ns),'turn':np.mean(t),'regimes':regs}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'possible',f.size,'positive_weight_days',int((w>0).sum()))
for h in (1,5,10,20):print('METRIC',json.dumps(met(h),default=float))
# Full reconstructable admitted-library correlation screen.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/v;orth=residual(acc,[trend]);spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});persist=r.le(r.quantile(.2,axis=1),axis=0).astype(float).rolling(60,min_periods=40).mean();pt=residual(persist,[trend]);short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});dep=residual(short-down,[down]);idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A});isk=residual(-idres.rolling(40,min_periods=30).skew(),[es,down,kurt,trend]);thr=vix.rolling(60,min_periods=40).quantile(.7);vp=residual(pd.DataFrame({a:r[a].where(vix>thr).rolling(60,min_periods=15).corr(peer[a].where(vix>thr)) for a in A}),[es,down,kurt,trend]);cthr=cny.rolling(60,min_periods=40).quantile(.7);cr=residual(pd.DataFrame({a:-beta(r[a],cny,60,15,cny>cthr) for a in A}),[du,es,down,trend]);vdown=residual(vd,[es,down,kurt,trend,vu]);vup=residual(vu,[es,down,kurt,trend])
lib={'ravmom':trend,'reversal':rev,'downside_corr':down,'autocorr':aut,'vixstress':vp,'cnyres':cr,'asym':asym,'peerchange':dep,'spx':spx,'kurtosis':kurt,'expected_shortfall':es,'upside_corr':upinv,'dxyup':du,'dxydown':dd,'idioskew':isk,'vixup':vup,'vixdown':vdown}
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan;print('LIB',n,'rho',rho,'cells',len(z))
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
``` 
