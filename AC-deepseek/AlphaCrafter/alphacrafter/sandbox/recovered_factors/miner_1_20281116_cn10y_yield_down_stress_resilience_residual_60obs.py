"""One idea: CN10Y yield-down shock resilience residual, 60 observations."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-11-15')
def ser(a,index=False):
 d=(get_index_daily_data(a,5000) if index else get_stock_daily_data(a,5000)).copy();d['date']=pd.to_datetime(d.date);return pd.to_numeric(d.set_index('date').loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:ser(a) for a in A});r=p.pct_change();idx=p.index
def mac(a):return ser(a,True).pct_change().reindex(idx)
def beta(x,y,mask=None,w=60,minp=15):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(w,min_periods=minp).cov(y)/y.rolling(w,min_periods=minp).var().replace(0,np.nan)
def residual(y,cs):
 o=y*np.nan
 for d in idx:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A}); vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol
dx=mac('DXY');vix=mac('VIX');cn10=r.CN10Y
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
dxyup=pd.DataFrame({a:-beta(r[a],dx,dx>0,40,12) for a in A})
# Higher: stronger conditional return loading when Chinese yields fall (rate-cut/liquidity shock sessions).
shock=cn10<cn10.rolling(60,min_periods=30).quantile(.30)
raw=pd.DataFrame({a:beta(r[a],cn10,shock,60,15) for a in A})
f=residual(raw,[dxyup,es,down,trend])
def metric(h):
 fw=p.shift(-h)/p-1;xs=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:xs.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs)); sd=x.std(); reg={}
 for name,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask];reg[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std(),'hit':(q>0).mean()}
 tr=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:tr.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(tr)),'regimes':reg}
print('FACTOR cn10y_yield_down_stress_resilience_residual_60obs');print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h),default=float))
# Active library only: exclude files explicitly deprecated.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol;orth=residual(acc,[trend])
spx=pd.DataFrame({a:-beta(r[a],r.SPX,None,20,15) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));kurt=-r.rolling(40,min_periods=30).kurt();upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0,60,15)-beta(r[a],peer[a],peer[a]>0,60,15) for a in A});short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A});dep=residual(short-down,[down]);idio=pd.DataFrame({a:r[a]-beta(r[a],peer[a],None,20,15)*peer[a] for a in A});isk=residual(-idio.rolling(40,min_periods=30).skew(),[es,down,kurt,trend]);vs=vix>vix.rolling(60,min_periods=30).quantile(.70);vr=pd.DataFrame({a:r[a].where(vs).rolling(60,min_periods=15).corr(peer[a].where(vs)) for a in A});vf=residual(vr,[es,down,kurt,trend]);uc=mac('USDCNY');us=uc>uc.rolling(60,min_periods=30).quantile(.70);uf=residual(pd.DataFrame({a:-beta(r[a],uc,us,60,15) for a in A}),[dxyup,es,down,trend])
# miner_2 inverse residual beta-compression definition reconstructed as inverse residual recent-minus-long downside correlation.
comp=residual(-(short-down),[down])
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':aut,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spx,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':upinv,'negative_conditional_dxy_up_beta_40obs':dxyup,'asymmetric_peer_beta_resilience_40obs':asym,'residual_downside_peer_dependence_change_20_60':dep,'inverse_residual_downside_peer_beta_compression_20_60':comp,'vix_stress_peer_correlation_residual_60obs':vf,'usdcny_appreciation_stress_resilience_residual_60obs':uf}
mx=-1;who=''
for n,q in lib.items():
 z=pd.concat([f.stack().rename('f'),q.stack().rename('q')],axis=1).dropna();rho=z.f.corr(z.q,method='spearman');print('LIB',n,'rho',rho,'cells',len(z))
 if not np.isfinite(rho):raise RuntimeError('missing evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
