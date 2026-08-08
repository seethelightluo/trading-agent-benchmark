"""One idea: relative volatility compression, 20d versus 60d realized volatility."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-20'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A});r=p.pct_change();idx=p.index
v20=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=45).std()
# Higher score: current volatility compressed relative to its own medium-term baseline.
f=-(v20/v60)
def rollcond(x,y,w,sel,mode='beta',sgn=1,mn=10):
 o=np.full(len(x),np.nan);X=x.to_numpy(float);Y=y.to_numpy(float)
 for i in range(w-1,len(X)):
  xx=X[i-w+1:i+1];yy=Y[i-w+1:i+1];q=sel(yy)&np.isfinite(xx)&np.isfinite(yy);xx=xx[q];yy=yy[q]
  if len(xx)>=mn and np.std(yy,ddof=1)>0:o[i]=sgn*(np.corrcoef(xx,yy)[0,1] if mode=='corr' and np.std(xx,ddof=1)>0 else np.cov(xx,yy,ddof=1)[0,1]/np.var(yy,ddof=1))
 return pd.Series(o,index=idx)
def peer(kind,mode='corr'):
 return pd.DataFrame({a:rollcond(r[a],r.drop(columns=a).mean(axis=1),40,(lambda z:z<0) if kind=='down' else (lambda z:z>0),mode,1 if kind=='down' else -1,12) for a in A})
def met(h):
 fw=p.shift(-h)/p-1;ics=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:ics.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(ics)); sd=x.std(ddof=1); regs={}
 for n,m in [('2020_2022',x.index.year<=2022),('2023_2024',x.index.year.isin([2023,2024])),('2025_2026',x.index.year.isin([2025,2026])),('2027',x.index.year==2027)]:
  y=x[m];regs[n]={'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit':float((y>0).mean())}
 tos=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic_dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(tos)),'regimes':regs}
print('FACTOR relative_volatility_compression_20_60obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H:print('METRIC',json.dumps(met(h),sort_keys=True))
trend=(p/p.shift(20)-1)/v20;rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/v20
orth=pd.DataFrame(index=idx,columns=A,dtype=float)
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
def macro(n):return pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
dxy=macro('DXY');vix=macro('VIX');spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A});dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A});part=pd.DataFrame({a:np.log(rd(a).volume.astype(float)/rd(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A});kurt=-r.rolling(40,min_periods=30).kurt();es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v20[a] for a in A});va=pd.DataFrame({a:rollcond(r[a],vix,60,lambda z:z>0,'beta',1)-rollcond(r[a],vix,60,lambda z:z<0,'beta',1) for a in A});down=peer('down');up=peer('up');downb=peer('down','beta');upb=peer('up','beta')
lib={'risk_adjusted_trend':trend,'ravmom_20obs':trend,'volnorm_reversal':rev,'inverse_realized_volatility':-v20,'relative_volume_participation':part,'orthogonal_trend_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asymmetric_shock_beta':va,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':down,'inverse_upside_peer_correlation':up,'negative_conditional_dxy_up_beta':pd.DataFrame({a:rollcond(r[a],dxy,40,lambda z:z>0,'beta',-1) for a in A}),'positive_conditional_dxy_down_beta':pd.DataFrame({a:rollcond(r[a],dxy,40,lambda z:z<0,'beta',-1) for a in A}),'asymmetric_peer_beta_resilience':downb+upb}
mx=-1;who=''; ev={}
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');ev[n]={'rho':None if pd.isna(q) else float(q),'common_cells':len(z)};print('DIRECT_LIB',n,'rho',q,'cells',len(z))
 if pd.isna(q):mx=np.nan
 elif not pd.isna(mx) and abs(q)>mx:mx=abs(q);who=n
print('DIRECT_MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if not pd.isna(mx) else 'MISSING_EVIDENCE');print('DIRECT_EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
