"""One idea: US10Y yield-up shock resilience residual, 60 observations."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-10-18')
def px(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:px(a) for a in A});r=p.pct_change();idx=p.index
def mac(a):
 d=get_index_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.loc[:END,'close'],errors='coerce').pct_change().reindex(idx)
def beta(x,y,mask=None,w=60,minp=15):
 if mask is not None:x=x.where(mask);y=y.where(mask)
 return x.rolling(w,min_periods=minp).cov(y)/y.rolling(w,min_periods=minp).var().replace(0,np.nan)
y= r.US10Y; dx=mac('DXY'); vix=mac('VIX'); shock=y>y.rolling(60,min_periods=30).quantile(.70)
# Higher is lower conditional loading to historically observed rate-rise shocks.
raw=pd.DataFrame({a:-beta(r[a],y,shock,60,15) for a in A})
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A});vol=r.rolling(20,min_periods=15).std();trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda q:np.mean(q[q<=np.quantile(q,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
dxyup=pd.DataFrame({a:-beta(r[a],dx,dx>0,40,12) for a in A})
def residual(y, controls):
 out=y*np.nan
 for d in idx:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(controls)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:].values];out.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
f=residual(raw,[dxyup,es,down,trend])
def metric(h):
 fw=p.shift(-h)/p-1;rows=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows)); sd=x.std(); reg={}
 for n,m in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('latest_120',np.arange(len(x))>=len(x)-120)]:
  q=x[m];reg[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan,'hit':(q>0).mean()}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'ic_se':sd/np.sqrt(len(x)),'mean_instruments':np.mean(ns),'turnover_10d':np.mean(turn),'regimes':reg}
print('FACTOR us10y_yield_up_stress_resilience_residual_60obs')
print('VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'of',f.size,'coverage',f.count().sum()/f.size)
for h in [1,5,10,20]:print('METRIC',json.dumps(metric(h),default=float))
# Reconstruct all current effective-library signals using their published definitions.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol; orth=residual(acc,[trend])
spx=pd.DataFrame({a:-beta(r[a],r.SPX,None,20,15) for a in A});aut=r.rolling(20,min_periods=15).corr(r.shift(1));kurt=-r.rolling(40,min_periods=30).kurt();upinv=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0,60,15)-beta(r[a],peer[a],peer[a]>0,60,15) for a in A})
short=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(20,min_periods=8).corr(peer[a].where(peer[a]<0)) for a in A}); dep=residual(short-down,[down]); idio=pd.DataFrame({a:r[a]-beta(r[a],peer[a],None,60,30)*peer[a] for a in A});isk=residual(-idio.rolling(40,min_periods=30).skew(),[es,down,kurt,trend])
vs=vix>vix.rolling(60,min_periods=30).quantile(.70); vr=pd.DataFrame({a:r[a].where(vs).rolling(60,min_periods=15).corr(peer[a].where(vs)) for a in A});vf=residual(vr,[es,down,kurt,trend])
# relative volume participation and inverse correlation-concentration definitions.
volu=pd.DataFrame({a:pd.to_numeric(get_stock_daily_data(a,5000).set_index('date').assign(date=lambda z:pd.to_datetime(z.index)).loc[:END,'volume'],errors='coerce') for a in A}).reindex(idx); rv=volu/volu.rolling(20,min_periods=15).mean()
conc=pd.DataFrame(index=idx,columns=A,dtype=float)
for d in idx:
 q=r.loc[:d].tail(40)
 if len(q)>=30:
  c=q.corr()
  for a in A:conc.loc[d,a]=-c.loc[a].drop(a).std()
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':aut,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spx,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':upinv,'negative_conditional_dxy_up_beta_40obs':dxyup,'positive_conditional_dxy_down_beta_40obs':pd.DataFrame({a:beta(r[a],dx,dx<0,40,12) for a in A}),'asymmetric_peer_beta_resilience_40obs':asym,'residual_downside_peer_dependence_change_20_60':dep,'vix_stress_peer_correlation_residual_60obs':vf,'orthogonal_inverse_idiosyncratic_skewness_40obs':isk,'relative_volume_participation_20d':rv,'inverse_cross_asset_correlation_concentration_40obs':conc,'usdcny_appreciation_stress_resilience_residual_60obs':None}
cn=mac('USDCNY'); cs=cn>cn.rolling(60,min_periods=30).quantile(.70); cr=pd.DataFrame({a:-beta(r[a],cn,cs,60,15) for a in A});lib['usdcny_appreciation_stress_resilience_residual_60obs']=residual(cr,[dxyup,es,down,trend])
mx=-1;who=''
for n,q in lib.items():
 z=pd.concat([f.stack().rename('f'),q.stack().rename('q')],axis=1).dropna();rho=z.f.corr(z.q,method='spearman')
 print('LIB',n,'rho',rho,'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX',mx,who)
