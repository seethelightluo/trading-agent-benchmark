"""One candidate only: efficient validation and complete admitted-library correlation evidence."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-09-22'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); idx=r.index
def conditional_beta(y, condition, window, sign=1):
 out=pd.DataFrame(index=idx,columns=A,dtype=float)
 for a in A:
  x=r[a]
  # Small loops are intentional: exact rolling conditional sample selection.
  for i in range(window-1,len(idx)):
   xx=x.iloc[i-window+1:i+1]; yy=y.iloc[i-window+1:i+1]; m=condition(yy)
   if m.sum()>=12 and yy[m].var()>0: out.iloc[i,out.columns.get_loc(a)]=sign*xx[m].cov(yy[m])/yy[m].var()
 return out
def peer_cond(kind, corr=True):
 out=pd.DataFrame(index=idx,columns=A,dtype=float)
 for a in A:
  y=r.drop(columns=a).mean(axis=1); x=r[a]
  for i in range(39,len(idx)):
   xx=x.iloc[i-39:i+1]; yy=y.iloc[i-39:i+1]; m=(yy<0) if kind=='down' else (yy>0)
   if m.sum()>=12 and (corr or yy[m].var()>0):
    out.iloc[i,out.columns.get_loc(a)]=(1 if kind=='down' else -1)*(xx[m].corr(yy[m]) if corr else xx[m].cov(yy[m])/yy[m].var())
 return out
# Candidate: inverse asset loss conditional on worst peer-quintile days, standardized by 20d vol.
f=pd.DataFrame(index=idx,columns=A,dtype=float)
for a in A:
 peer=r.drop(columns=a).mean(axis=1)
 for i in range(39,len(idx)):
  x=r[a].iloc[i-39:i+1]; y=peer.iloc[i-39:i+1]; m=y<=y.quantile(.2); s=x.std(ddof=1)
  if m.sum()>=8 and s>0:f.iloc[i,f.columns.get_loc(a)]=-x[m].mean()/s
# Metrics.
def metric(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('ret')],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.f.corr(z.ret,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(ddof=1); regs={}
 for nm,mask in [('2020_2022',x.index.year<=2022),('2023_2024',x.index.year.isin([2023,2024])),('2025_2026',x.index.year.isin([2025,2026])),('2027',x.index.year==2027)]:
  q=x[mask];regs[nm]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean())}
 ts=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic_dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(ts)),'regimes':regs}
print('FACTOR inverse_peer_tail_loss_loading_40obs'); print('visible_through',END.date(),'period',p.index.min().date(),p.index.max().date(),'assets',len(A));print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H: print('METRIC',json.dumps(metric(h),sort_keys=True))
# Reconstruct all 15 admitted signals from their stated definitions.
trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=idx,columns=A,dtype=float)
for d in idx:
 z=pd.concat([acc.loc[d].rename('a'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.a-np.c_[np.ones(len(z)),z.t]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.t],z.a,rcond=None)[0]
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A})
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A})
part=pd.DataFrame({a:np.log(rd(a).volume.astype(float)/rd(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A}).replace([np.inf,-np.inf],np.nan)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
va=pd.DataFrame(index=idx,columns=A,dtype=float)
for a in A:
 for i in range(59,len(idx)):
  x=r[a].iloc[i-59:i+1]; y=vix.iloc[i-59:i+1]; u=y>0; d=y<0
  if u.sum()>=10 and d.sum()>=10 and y[u].var()>0 and y[d].var()>0: va.iloc[i,va.columns.get_loc(a)]=x[u].cov(y[u])/y[u].var()-x[d].cov(y[d])/y[d].var()
lib={'risk_adjusted_trend':trend,'ravmom_20obs':trend,'volnorm_reversal':rev,'inverse_realized_volatility':-vol,'relative_volume_participation':part,'orthogonal_trend_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asymmetric_shock_beta':va,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':peer_cond('down'),'inverse_upside_peer_correlation':peer_cond('up'),'negative_conditional_dxy_up_beta':conditional_beta(dxy,lambda y:y>0,40,-1),'positive_conditional_dxy_down_beta':conditional_beta(dxy,lambda y:y<0,40,-1),'asymmetric_peer_beta_resilience':peer_cond('down',False)-(-peer_cond('up',False))}
out={}; mx=-1;who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman');out[n]={'rho':None if pd.isna(rho) else float(rho),'common_cells':len(z)};print('DIRECT_LIB',n,'rho',rho,'cells',len(z))
 if pd.isna(rho): mx=np.nan
 elif not pd.isna(mx) and abs(rho)>mx:mx=abs(rho);who=n
print('DIRECT_MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if not pd.isna(mx) else 'MISSING_EVIDENCE');print('DIRECT_EVIDENCE_JSON',json.dumps(out,sort_keys=True))
