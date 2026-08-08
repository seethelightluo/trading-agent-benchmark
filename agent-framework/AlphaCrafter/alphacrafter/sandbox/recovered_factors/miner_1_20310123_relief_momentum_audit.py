import pandas as pd,numpy as np,glob,json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 f=glob.glob('../persistent/stock_data/'+a+'.csv')+glob.glob('../persistent/stock_data/'+a+'/*.csv')
 d=pd.read_csv(f[0]);d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float)
def macro(a):
 d=pd.read_csv('../persistent/index_data/'+a+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float)
p=pd.concat({a:load(a) for a in A},axis=1).sort_index();r=np.log(p).diff();vol=r.rolling(20).std(); peer=r.sub(r.mean(axis=1),axis=0)
v=np.log(macro('VIX')).diff().reindex(p.index);d=np.log(macro('DXY')).diff().reindex(p.index)
mask=(v.rolling(5).sum()<0)&(d.rolling(5).sum()<0)
f=r.rolling(20).sum();f=f.sub(f.mean(axis=1),axis=0).where(mask)
def beta(x,y,w=40,cond=None):
 if cond is not None:x=x.where(cond);y=y.where(cond)
 return x.rolling(w,min_periods=12).cov(y)/y.rolling(w,min_periods=12).var()
def resid(x,cs):
 o=x*np.nan
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('y')]+[c.loc[dt].rename(str(j)) for j,c in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
trend=(p/p.shift(20)-1)/vol; es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
vu=pd.DataFrame({a:r[a].where(v>0).rolling(40,min_periods=12).mean()/vol[a] for a in A}); vd=pd.DataFrame({a:r[a].where(v<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
du=pd.DataFrame({a:-beta(r[a],d,40,d>0) for a in A});dd=pd.DataFrame({a:beta(r[a],d,40,d<0) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt(); rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib={'ravmom':trend,'reversal':rev,'downside_peer':down,'trend_accel':resid((p/p.shift(20)-p.shift(20)/p.shift(60))/vol,[trend]),'neg_spx':pd.DataFrame({a:-beta(r[a],r.SPX,20) for a in A}),'es':es,'kurt':kurt,'up_res':resid(vu,[es,down,kurt,trend]),'down_res':resid(vd,[es,down,kurt,trend,vu]),'dxy_up':du,'dxy_down':dd,'asym':beta(r,peer,40,peer<0)-beta(r,peer,40,peer>0),'vix_transition':resid(vu-vd,[es,down,kurt,trend,du,dd]),'relief_fragility':resid(r.where(v<0).rolling(80,min_periods=24).mean()/vol-r.where(v<0).rolling(20,min_periods=8).mean()/vol,[vd,vu,es,down,kurt,trend,du,dd])}
# metrics
for h in [1,5,10,20]:
 q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q),'meanN',15)
mx=0;who=''
for n,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
 print('LIB',n,round(rho,6),len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,who,'coverage',f.count().sum()/f.size,'visible',p.index.max().date())
