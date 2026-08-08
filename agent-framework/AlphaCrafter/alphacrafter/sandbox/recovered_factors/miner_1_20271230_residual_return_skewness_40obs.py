"""One idea: 40-session residual return skewness, separating asymmetric upside tails from tail-risk and trend."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2027-12-29')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); idx=p.index; r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def skew(x):
 x=x[np.isfinite(x)]; return np.mean(((x-x.mean())/x.std(ddof=0))**3) if len(x)>=30 and x.std(ddof=0)>0 else np.nan
raw=r.rolling(40,min_periods=30).apply(skew,raw=True)
# Residualize cross-sectionally each day against admitted tail-level variables to isolate tail shape.
kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v20[a] for a in A}); trend=(p/p.shift(20)-1)/v20
f=raw*np.nan
for d in idx:
 z=pd.concat([raw.loc[d].rename('y'),kurt.loc[d].rename('k'),es.loc[d].rename('e'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['k','e','t']]];f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def rollcond(x,y,w,sel,mode='beta',sgn=1,mn=12):
 out=np.full(len(x),np.nan);X=x.to_numpy(float);Y=y.to_numpy(float)
 for i in range(w-1,len(X)):
  xx=X[i-w+1:i+1];yy=Y[i-w+1:i+1];q=sel(yy)&np.isfinite(xx)&np.isfinite(yy);xx=xx[q];yy=yy[q]
  if len(xx)>=mn and np.std(yy,ddof=1)>0: out[i]=sgn*(np.corrcoef(xx,yy)[0,1] if mode=='corr' and np.std(xx,ddof=1)>0 else np.cov(xx,yy,ddof=1)[0,1]/np.var(yy,ddof=1))
 return pd.Series(out,index=idx)
def beta_df(y,sel,w=40,sgn=1): return pd.DataFrame({a:rollcond(r[a],y,w,sel,'beta',sgn) for a in A})
def corr_df(sel,sgn=1): return pd.DataFrame({a:rollcond(r[a],peer[a],40,sel,'corr',sgn) for a in A})
def metrics(h):
 fw=p.shift(-h)/p-1; rows=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows)); sd=x.std(ddof=1); regs={}
 for n,years in {'2020':[2020],'2021_2022':[2021,2022],'2023_2024':[2023,2024],'2025_2026':[2025,2026],'2027':[2027]}.items():
  q=x[x.index.year.isin(years)];regs[n]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 and q.std(ddof=1)>0 else None,'hit':float((q>0).mean()) if len(q) else None}
 to=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(to)),'regimes':regs}
print('FACTOR residual_return_skewness_40obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h),sort_keys=True))
# Reconstruct every non-deprecated admitted signal for binding pooled-cell Spearman gate.
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/v20;orth=acc*np.nan
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
spxb=beta_df(r.SPX,lambda z:np.full(len(z),True),20,-1)
dxy=pd.to_numeric(get_index_daily_data('DXY',5000).set_index('date').assign(date=lambda x:pd.to_datetime(x.index)).loc[:END,'close'],errors='coerce');dxy.index=pd.to_datetime(dxy.index);dxy=dxy.pct_change().reindex(idx)
vix=pd.to_numeric(get_index_daily_data('VIX',5000).set_index('date').assign(date=lambda x:pd.to_datetime(x.index)).loc[:END,'close'],errors='coerce');vix.index=pd.to_datetime(vix.index);vix=vix.pct_change().reindex(idx)
dxyb=beta_df(dxy,lambda z:np.full(len(z),True),20); va=beta_df(vix,lambda z:z>0,60)-beta_df(vix,lambda z:z<0,60)
part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
down=corr_df(lambda z:z<0);up=corr_df(lambda z:z>0,-1);du=beta_df(dxy,lambda z:z>0,40,-1);dd=beta_df(dxy,lambda z:z<0,40);downb=beta_df(peer.iloc[:,0],lambda z:z<0) # overwritten next
# exact per-asset asymmetric peer conditional beta
asym=pd.DataFrame({a:rollcond(r[a],peer[a],40,lambda z:z<0,'beta',1)-rollcond(r[a],peer[a],40,lambda z:z>0,'beta',1) for a in A})
autoc=r.rolling(20,min_periods=15).corr(r.shift(1)); asympeer=pd.DataFrame({a:rollcond(r[a],peer[a],40,lambda z:z<0,'beta',1)-rollcond(r[a],peer[a],40,lambda z:z>0,'beta',-1) for a in A})
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':autoc,'relative_volume_participation_20d':part,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spxb,'dxy_beta_20obs':dxyb,'vix_asymmetric_shock_beta_60obs':va,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':up,'negative_conditional_dxy_up_beta_40obs':du,'positive_conditional_dxy_down_beta_40obs':dd,'asymmetric_peer_beta_resilience_40obs':asympeer}
ev={};mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');ev[n]={'rho':None if pd.isna(q) else float(q),'common_cells':len(z)};print('LIB',n,'rho',q,'cells',len(z))
 if pd.isna(q): mx=np.nan
 elif not pd.isna(mx) and abs(q)>mx:mx=abs(q);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if not pd.isna(mx) else 'MISSING_EVIDENCE');print('LIBRARY_EVIDENCE',json.dumps(ev,sort_keys=True))
