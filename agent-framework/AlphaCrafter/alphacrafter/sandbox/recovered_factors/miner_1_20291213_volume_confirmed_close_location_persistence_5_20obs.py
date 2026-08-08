"""Single-factor test: volume-confirmed close-location persistence.
The factor averages the signed close location over five native bars, weighted by
contemporaneous abnormal volume. It tests whether directional intraday closes
with participation predict cross-asset forward returns."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-12-12')
def ld(a,c='close',idx=False):
 p=('../persistent/index_data/' if idx else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,c].astype(float)
p={a:ld(a) for a in A}; cp=pd.DataFrame(p); r=cp.pct_change(); med=r.median(axis=1)
hi=pd.DataFrame({a:ld(a,'high') for a in A}); lo=pd.DataFrame({a:ld(a,'low') for a in A}); v=pd.DataFrame({a:ld(a,'volume').replace(0,np.nan) for a in A})
clv=((2*cp-hi-lo)/(hi-lo).replace(0,np.nan)).clip(-1,1)
# one construction: a close near high/low only contributes in proportion to its own volume surprise
abvol=np.log(v/v.rolling(20,min_periods=15).mean()).clip(-3,3)
f=(clv*abvol).rolling(5,min_periods=4).mean()
def rc(x,y,n=20,mi=15): return x.rolling(n,min_periods=mi).corr(y)
def beta(x,y,n,mi,mask=None):
 if mask is not None: x=x.where(mask); y=y.where(mask)
 return x.rolling(n,min_periods=mi).cov(y)/y.rolling(n,min_periods=mi).var()
def csres(x,z):
 out=x*np.nan
 for t in x.index:
  q=pd.concat([x.loc[t],z.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,1].var()>0:
   out.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 return out
def cmean(x,m,w,mi): return x.where(m).rolling(w,min_periods=mi).mean()
s20=r.rolling(20,min_periods=15).std(); s60=r.rolling(60,min_periods=45).std()
fast=(cp/cp.shift(20)-1)/s20; slow=(cp/cp.shift(60)-1)/s60
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':abvol,'realized_volatility_20obs':s20,'volnorm_reversal_5obs':-(cp/cp.shift(5)-1)/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.apply(lambda x:rc(x,x.shift())),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().apply(lambda x:rc(x,x.shift())),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean())}
b=pd.DataFrame({a:beta(r[a],med,60,45) for a in A}); dn=med<0; up=~dn
ca=pd.DataFrame({a:rc(r[a].where(dn),med.where(dn),60,10)-rc(r[a].where(up),med.where(up),60,10) for a in A}); lib['correlation_asymmetry_60obs']=ca
res=pd.DataFrame({a:r[a]-(r[a].rolling(60,min_periods=45).mean()+b[a]*(med-med.rolling(60,min_periods=45).mean())) for a in A});lib['residual_downside_semivol_share_60obs']=res.clip(upper=0).pow(2).rolling(60,min_periods=45).mean().pow(.5)/res.pow(2).rolling(60,min_periods=45).mean().pow(.5);lib['vol_orthogonal_median_beta_60obs']=csres(b,s20)
D=r.std(axis=1);lib['dispersion_sensitivity_20obs']=r.apply(lambda x:rc(x,D));lib['excess_downside_beta_ca_orthogonal_60obs']=csres(pd.DataFrame({a:beta(r[a],med,60,10,dn)-b[a] for a in A}),ca)
vix=ld('VIX',idx=True).pct_change();lib['adaptive_vix_relief_beta_change_25_60obs']=pd.DataFrame({a:beta(r[a],vix,25,8,vix<0)-beta(r[a],vix,60,18,vix<0) for a in A});lib['vix_shock_relief_beta_asymmetry_60obs']=pd.DataFrame({a:beta(r[a],vix,60,12,vix>0)-beta(r[a],vix,60,12,vix<0) for a in A}); tail=vix.abs()>vix.abs().rolling(60,min_periods=45).quantile(.8);lib['vix_tail_lagged_response_persistence_60obs']=pd.DataFrame({a:beta(r[a],vix.shift(),60,12,tail.shift()) for a in A})
dxy=ld('DXY',idx=True).pct_change();mt=med.rolling(20,min_periods=15).mean();lib['dxy_median_trend_regime_beta_spread_60obs']=pd.DataFrame({a:beta(r[a],dxy,60,12,mt>0)-beta(r[a],dxy,60,12,mt<=0) for a in A});rv=s20/s20.rolling(60,min_periods=45).median();lib['dxy_relative_vol_regime_beta_spread_60obs']=pd.DataFrame({a:beta(r[a],dxy,60,12,rv[a]>1)-beta(r[a],dxy,60,12,rv[a]<=1) for a in A}); dtail=dxy.abs()>dxy.abs().rolling(60,min_periods=45).quantile(.8);lib['dxy_shock_lagged_response_persistence_60obs']=pd.DataFrame({a:beta(r[a],dxy.shift(),60,12,dtail.shift()) for a in A})
dva=pd.DataFrame({a:np.log(cmean(v[a],r[a]<0,60,15)/cmean(v[a],r[a]>=0,60,15)) for a in A});lib['downside_volume_participation_asymmetry_60obs']=dva;lib['inverted_downside_volume_participation_acceleration_20_60obs']=pd.DataFrame({a:dva[a]-np.log(cmean(v[a],r[a]<0,20,8)/cmean(v[a],r[a]>=0,20,8)) for a in A}); rg=(hi-lo)/cp;state=np.sign(np.log(s20.median(axis=1)/s20.median(axis=1).rolling(60,min_periods=45).median()));lib['inverted_dispersion_regime_range_state_20_60obs']=-np.log(rg.rolling(20,min_periods=15).mean()/rg.rolling(60,min_periods=45).mean()).mul(state,axis=0)
print('CANDIDATE volume_confirmed_close_location_persistence_5_20obs cutoff',END.date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'of',len(f)*15)
best=None
for h in [1,5,10,20]:
 y=cp.shift(-h)/cp-1; xs=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: xs.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 x=pd.Series(dict(xs)); ic=x.mean(); ir=ic/x.std(ddof=1); print('H',h,'dates',len(x),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((x>0).mean(),4),'mean_n',round(np.mean(ns),2),'coverage',round(f.notna().mean().mean(),4))
 if best is None or abs(ic*ir)>abs(best[1].mean()*(best[1].mean()/best[1].std(ddof=1))):best=(h,x)
h,x=best; ranks=f.rank(axis=1); turn=1-ranks.corrwith(ranks.shift(),axis=1,method='spearman'); cors={}
for name,z in lib.items():
 vv=[]
 for t in f.index:
  q=pd.concat([f.loc[t],z.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:vv.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 cors[name]=np.max(np.abs(vv)) if vv else np.nan
print('BEST',h,'turnover',round(turn.mean(),4),'max_abs_library_correlation',round(max(cors.values()),6),'against',max(cors,key=cors.get))
for name,sub in [('2020_21',x[x.index<'2022-01-01']),('2022_23',x[(x.index>='2022-01-01')&(x.index<'2024-01-01')]),('2024_25',x[(x.index>='2024-01-01')&(x.index<'2026-01-01')]),('2026_current',x[x.index>='2026-01-01'])]:print('REGIME',name,'dates',len(sub),'IC',round(sub.mean(),6),'ICIR',round(sub.mean()/sub.std(ddof=1),6),'hit',round((sub>0).mean(),4))
