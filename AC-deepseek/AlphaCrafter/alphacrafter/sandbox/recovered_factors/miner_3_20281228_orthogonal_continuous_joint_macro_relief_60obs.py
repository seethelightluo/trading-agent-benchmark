"""One candidate: orthogonal continuous joint macro-relief participation, full-history test."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-12-27')
def rd(a,ix=False):
 d=(get_index_daily_data(a,5000) if ix else get_stock_daily_data(a,5000)).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A});r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def bet(x,y,w=40,mp=12,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
def residual(y,controls):
 out=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(controls)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];out.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
vix=rd('VIX',True).pct_change().reindex(r.index);dxy=rd('DXY',True).pct_change().reindex(r.index)
trend=(p/p.shift(20)-1)/v
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});kurt=-r.rolling(40,min_periods=30).kurt()
vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/v[a] for a in A});vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/v[a] for a in A});du=pd.DataFrame({a:r[a].where(dxy>0).rolling(40,min_periods=12).mean()/v[a] for a in A});dd=pd.DataFrame({a:r[a].where(dxy<0).rolling(40,min_periods=12).mean()/v[a] for a in A})
# Smooth standardized relief score avoids sparse event intersection; high factor means strong normalized participation as VIX and DXY ease.
relief=(-(vix/vix.rolling(60,min_periods=40).std())-(dxy/dxy.rolling(60,min_periods=40).std())).clip(-4,4)
raw=pd.DataFrame({a:(r[a]*relief).rolling(60,min_periods=30).mean()/v[a] for a in A})
f=residual(raw,[es,down,kurt,trend,vu,vd,du,dd])
def met(h):
 fw=p.shift(-h)/p-1; q=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(q));sd=x.std(); reg={}
 for name,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('latest120',np.arange(len(x))>=len(x)-120)]:
  y=x[m];reg[name]={'dates':len(y),'ic':y.mean(),'icir':y.mean()/y.std()}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'se':sd/np.sqrt(len(x)),'n':np.mean(ns),'turn':np.mean(turn),'regimes':reg}
print('VISIBLE',END.date(),'assets',len(A),'dates',len(p),'cells',int(f.count().sum()),'possible',f.size)
for h in (1,5,10,20):print('METRIC',json.dumps(met(h),default=float))
# Conservative mandatory comparison: candidate is explicitly residualized against its closest related admitted VIX/DXY and defensive controls.
lib={'inverse_es':es,'downside_corr':down,'inverse_kurtosis':kurt,'risk_adj_trend':trend,'vix_up_raw':vu,'vix_down_raw':vd,'dxy_up_raw':du,'dxy_down_raw':dd}
for n,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).dropna();print('LIB',n, z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))
