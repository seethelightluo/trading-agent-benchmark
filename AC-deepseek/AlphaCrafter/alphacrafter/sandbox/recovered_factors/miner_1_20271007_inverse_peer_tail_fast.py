"""One idea: inverse peer-tail loss loading, exact fast reconstruction for library decorrelation."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-06'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A}); r=p.pct_change(); idx=p.index; vol=r.rolling(20,min_periods=15).std()
def rollcond(x,y,w,select,mode,sign=1,minn=12):
 # exact rolling conditional covariance/correlation, numpy implementation
 o=np.full(len(x),np.nan); X=x.to_numpy(float);Y=y.to_numpy(float)
 for i in range(w-1,len(X)):
  xx=X[i-w+1:i+1]; yy=Y[i-w+1:i+1]; m=select(yy); xx=xx[m];yy=yy[m]; good=np.isfinite(xx)&np.isfinite(yy)
  xx=xx[good];yy=yy[good]
  if len(xx)>=minn and np.std(yy,ddof=1)>0:
   o[i]=sign*(np.corrcoef(xx,yy)[0,1] if mode=='corr' and np.std(xx,ddof=1)>0 else np.cov(xx,yy,ddof=1)[0,1]/np.var(yy,ddof=1))
 return pd.Series(o,index=idx)
def peer(kind,mode='corr'):
 q={}
 for a in A:
  y=r.drop(columns=a).mean(axis=1);q[a]=rollcond(r[a],y,40,(lambda z:z<0) if kind=='down' else (lambda z:z>0),mode,1 if kind=='down' else -1)
 return pd.DataFrame(q)
# Candidate definition
f={}
for a in A:
 x=r[a].to_numpy(float); y=r.drop(columns=a).mean(axis=1).to_numpy(float);o=np.full(len(x),np.nan)
 for i in range(39,len(x)):
  xx=x[i-39:i+1];yy=y[i-39:i+1]; good=np.isfinite(xx)&np.isfinite(yy);xx=xx[good];yy=yy[good]
  if len(xx)>=30:
   z=xx[yy<=np.quantile(yy,.2)]; s=np.std(xx,ddof=1)
   if len(z)>=8 and s>0:o[i]=-np.mean(z)/s
 f[a]=o
f=pd.DataFrame(f,index=idx)
def metrics(h):
 fw=p.shift(-h)/p-1; ic=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:ic.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(ic)); sd=x.std(ddof=1); reg={}
 for n,m in [('2020_2022',x.index.year<=2022),('2023_2024',x.index.year.isin([2023,2024])),('2025_2026',x.index.year.isin([2025,2026])),('2027',x.index.year==2027)]:
  q=x[m];reg[n]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean())}
 to=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic_dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(to)),'regimes':reg}
print('FACTOR inverse_peer_tail_loss_loading_40obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H:print('METRIC',json.dumps(metrics(h),sort_keys=True))
trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=idx,columns=A,dtype=float)
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A})
def macro(n):return pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
dxy=macro('DXY');vix=macro('VIX');dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A});part=pd.DataFrame({a:np.log(rd(a).volume.astype(float)/rd(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A});kurt=-r.rolling(40,min_periods=30).kurt();es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
va=pd.DataFrame({a:rollcond(r[a],vix,60,lambda z:z>0,'beta',1,10)-rollcond(r[a],vix,60,lambda z:z<0,'beta',1,10) for a in A})
down=peer('down');up=peer('up');downb=peer('down','beta');upb=peer('up','beta')
lib={'risk_adjusted_trend':trend,'ravmom_20obs':trend,'volnorm_reversal':rev,'inverse_realized_volatility':-vol,'relative_volume_participation':part,'orthogonal_trend_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asymmetric_shock_beta':va,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':down,'inverse_upside_peer_correlation':up,'negative_conditional_dxy_up_beta':pd.DataFrame({a:rollcond(r[a],dxy,40,lambda z:z>0,'beta',-1) for a in A}),'positive_conditional_dxy_down_beta':pd.DataFrame({a:rollcond(r[a],dxy,40,lambda z:z<0,'beta',-1) for a in A}),'asymmetric_peer_beta_resilience':downb+upb}
out={};mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');out[n]={'rho':None if pd.isna(rho) else float(rho),'common_cells':len(z)};print('DIRECT_LIB',n,'rho',rho,'cells',len(z))
 if pd.isna(rho):mx=np.nan
 elif not pd.isna(mx) and abs(rho)>mx:mx=abs(rho);who=n
print('DIRECT_MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if not pd.isna(mx) else 'MISSING_EVIDENCE');print('DIRECT_EVIDENCE_JSON',json.dumps(out,sort_keys=True))
