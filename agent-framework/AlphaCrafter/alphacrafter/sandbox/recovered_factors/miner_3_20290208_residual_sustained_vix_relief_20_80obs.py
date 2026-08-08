"""One candidate: residual multi-horizon VIX-relief resilience, validated point-in-time."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-02-07')
def rd(a,ix=False):
 d=(get_index_daily_data(a,5000) if ix else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A});r=p.pct_change(); vol=r.rolling(20,min_periods=15).std();peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,mp=12,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:o.loc[d,z.index]=z.y-np.c_[np.ones(len(z)),z.iloc[:,1:]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1:]],z.y,rcond=None)[0]
 return o
trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
vix=rd('VIX',True).pct_change().reindex(r.index); dxy=rd('DXY',True).pct_change().reindex(r.index)
vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A});vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A});dd=pd.DataFrame({a:beta(r[a],dxy,40,12,dxy<0) for a in A})
# Candidate rewards sustained relief performance relative to immediate relief performance; residualization makes it incremental to the admitted 40d relief signal.
short=pd.DataFrame({a:r[a].where(vix<0).rolling(20,min_periods=8).mean()/vol[a] for a in A})
long=pd.DataFrame({a:r[a].where(vix<0).rolling(80,min_periods=24).mean()/vol[a] for a in A})
raw=long-short
f=resid(raw,[vd,vu,es,down,kurt,trend,du,dd])
def metrics(h):
 fw=p.shift(-h)/p-1; q=[]; nn=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));nn.append(len(z))
 x=pd.Series(dict(q));sd=x.std(); regs={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[mask];regs[name]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std() if y.std() else np.nan}
 turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'mean_instruments':np.mean(nn),'turnover_10d':np.mean(turns),'regimes':regs}
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'possible',f.size,'vix_down_days',int((vix<0).sum()))
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h),default=float))
# Evidence against all active/admitted signal constructions available in this research family.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=resid((p/p.shift(20)-p.shift(20)/p.shift(60))/vol,[trend]); spx=pd.DataFrame({a:-beta(r[a],r.SPX) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,12,peer[a]<0)-beta(r[a],peer[a],40,12,peer[a]>0) for a in A});idres=pd.DataFrame({a:r[a]-beta(r[a],peer[a])*peer[a] for a in A});isk=resid(-idres.rolling(40,min_periods=30).skew(),[es,down,kurt,trend]);vdown=resid(vd,[es,down,kurt,trend,vu]);vup=resid(vu,[es,down,kurt,trend]);lib={'trend':trend,'reversal':rev,'downside_corr':down,'autocorr':aut,'acceleration':acc,'spx':spx,'kurtosis':kurt,'expected_shortfall':es,'upside_corr':up,'dxyup':du,'dxydown':dd,'idioskew':isk,'vixup':vup,'vixdown':vdown,'asym':asym}
mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,'rho',rho,'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
