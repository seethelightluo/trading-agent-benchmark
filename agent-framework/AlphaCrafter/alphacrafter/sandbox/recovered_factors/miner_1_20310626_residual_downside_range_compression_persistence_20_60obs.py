"""One idea: residual-downside range compression persistence (60 observations) and full active-library novelty audit."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-06-25'); W=60
def ld(a,col='close',root='../persistent/stock_data'):
 return pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:ld(a) for a in A}); vo=pd.DataFrame({a:ld(a,'volume').replace(0,np.nan) for a in A});hi=pd.DataFrame({a:ld(a,'high') for a in A});lo=pd.DataFrame({a:ld(a,'low') for a in A});r=p.pct_change(fill_method=None);m=r.median(axis=1);v=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=45).std()
def macro(a):return ld(a,root='../persistent/index_data').reindex(p.index).pct_change(fill_method=None)
def beta(x,y,w=60,cond=None,mp=12):
 o={}
 for a in A:
  xx=x[a];yy=y
  if cond is not None:
   c=cond[a] if isinstance(cond,pd.DataFrame) else cond;xx=xx.where(c);yy=yy.where(c)
  o[a]=xx.rolling(w,min_periods=mp).cov(yy)/yy.rolling(w,min_periods=mp).var()
 return pd.DataFrame(o)
def orth(x,z):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  q=pd.concat([x.loc[t],z.loc[t]],axis=1).dropna()
  if len(q)>=8:
   b=np.polyfit(q.iloc[:,1],q.iloc[:,0],1);o.loc[t]=x.loc[t]-(b[1]+b[0]*z.loc[t])
 return o
b=beta(r,m,60,mp=45);res=r-b.mul(m,axis=0)
# Higher value: residual-downside days have recently become less range-expansive
# relative to their own longer conditional baseline (selling pressure is compressing).
rng=(hi-lo).abs()/p
cond=res.shift(1)<0
s20=rng.where(cond).rolling(20,min_periods=5).mean();s60=rng.where(cond).rolling(60,min_periods=12).mean()
f=-np.log(s20/s60)
print('FACTOR residual_downside_range_compression_persistence_20_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H);z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns);print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),5),'mean_n',round(np.mean(ns),2),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=max(R,key=lambda h:abs(R[h][0].mean()*R[h][0].mean()/R[h][0].std(ddof=1));z,ds,_=R[best];print('SELECTED',best)
for n,a,c in [('2026_29','2026-01-01','2029-12-31'),('2030_31_ytd','2030-01-01',END)]:
 x=z[(ds>=a)&(ds<=c)];print('REGIME',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
rnk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER',round(np.mean(turn),6),'comparisons',len(turn),'CONCENTRATION_MEDIAN_IQR',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Reconstruct all 30 admitted signals as in prior audit.
vix=macro('VIX');dxy=macro('DXY');negday=r<0;ca=pd.DataFrame({a:r[a].where(m<0).rolling(60,min_periods=12).corr(m.where(m<0))-r[a].where(m>=0).rolling(60,min_periods=12).corr(m.where(m>=0)) for a in A});db=beta(r,m,60,m<0)-b
D20=np.log(vo.where(negday).rolling(20,min_periods=5).mean()/vo.where(~negday).rolling(20,min_periods=5).mean());D60=np.log(vo.where(negday).rolling(60,min_periods=12).mean()/vo.where(~negday).rolling(60,min_periods=12).mean());cs=v.median(axis=1);base=np.log(rng.rolling(20,min_periods=15).mean()/rng.rolling(60,min_periods=45).mean());stress=(-m.shift(1)/m.rolling(60,min_periods=45).std()).clip(-4,4);abvol=np.log(vo/vo.rolling(20,min_periods=15).median());posres=res.where(res>0);upsk=posres.pow(3).rolling(60,min_periods=30).mean()/np.sqrt(posres.pow(2).rolling(60,min_periods=30).mean()).pow(3);clv=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1);priorclv=clv.where(res.shift(1)<0).rolling(60,min_periods=30).mean();dispwt=(r.std(axis=1)/r.std(axis=1).rolling(60,min_periods=45).median()).clip(.25,4);latest=-pd.DataFrame({a:(res[a]*dispwt).rolling(60,min_periods=45).corr((res[a]*dispwt).shift()) for a in A});neg=res.shift(1)<0;valid=res.notna()&res.shift(1).notna()
L={'residual_downside_close_location_recovery':priorclv,'realized_volatility':v,'volnorm_reversal':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'correlation_asymmetry':ca,'return_sign_balance':(r>0).rolling(20,min_periods=15).mean()-.5,'dispersion_sensitivity':pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r.std(axis=1)) for a in A}),'volatility_clustering':r.abs().rolling(20,min_periods=15).corr(r.abs().shift()),'adaptive_vix_relief':beta(r,vix,25,vix<0)-beta(r,vix,60,vix<0),'vix_shock_relief':beta(r,vix,60,vix>0)-beta(r,vix,60,vix<0),'dxy_trend_regime':beta(r,dxy,60,m.rolling(20,min_periods=15).median()>0)-beta(r,dxy,60,m.rolling(20,min_periods=15).median()<=0),'dxy_relativevol':beta(r,dxy,60,v.gt(v.rolling(60,min_periods=45).median()))-beta(r,dxy,60,v.le(v.rolling(60,min_periods=45).median())),'residual_downside_semivol':np.sqrt(res.clip(upper=0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(res.pow(2).rolling(60,min_periods=45).mean()),'vol_orthogonal_beta':orth(b,v),'inverse_residual_transition':-((neg&(res>0)).rolling(W,min_periods=45).sum()/neg.where(valid).rolling(W,min_periods=45).sum()),'downside_volume':D60,'inverted_downside_volume_acceleration':-(D20-D60),'inverted_dispersion_range':-base.mul(np.sign(np.log(cs/cs.rolling(60,min_periods=45).median())).replace(0,1),axis=0),'relative_volume':np.log(vo/vo.rolling(20,min_periods=15).mean()),'risk_adjusted_trend':(p/p.shift(20)-1)/v,'trend_acceleration':(p/p.shift(20)-1)/v-(p/p.shift(60)-1)/v60,'return_persistence':r.rolling(20,min_periods=15).corr(r.shift()),'directional_efficiency':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'liquidity_stress':np.log((r.abs()/vo).rolling(20,min_periods=15).mean()/(r.abs()/vo).rolling(60,min_periods=45).mean()),'dxy_shock_lagged':beta(r,dxy.where(dxy.abs()>dxy.rolling(60,min_periods=45).std()),60),'vix_tail_lagged':beta(r,vix.where(vix.abs()>vix.rolling(60,min_periods=45).std()),60),'common_stress_clv':beta(clv,stress,60,mp=30),'common_stress_volume':beta(abvol,stress,60,mp=30),'upside_tail_skew':upsk,'overnight_reversal':((p/p.shift(1)-1)*(p.shift(1)/p.shift(2)-1)<0).rolling(20,min_periods=15).mean(),'inverse_dispersion_resid_persistence':latest}
mx=-1
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'factor',who,'cells',cells,'signals_tested',len(L))
