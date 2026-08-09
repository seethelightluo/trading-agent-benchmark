"""One idea: cross-sectionally residualized short/medium volatility expansion (10/60d)."""
import pandas as pd, numpy as np, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-02-23')
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A}); r=p.pct_change(); idx=p.index
v10=r.rolling(10,min_periods=8).std(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=45).std()
raw=v10/v60
# On each date remove the cross-sectional component explained by current 20d volatility level.
f=raw*np.nan
for d in idx:
 z=pd.concat([raw.loc[d].rename('y'),np.log(v20.loc[d]).rename('x')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.x]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metrics(h):
 fw=p.shift(-h)/p-1; rows=[]; ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows)); sd=x.std(ddof=1); regs={}
 for n,yrs in {'2020_22':[2020,2021,2022],'2023_24':[2023,2024],'2025_26':[2025,2026],'2027':[2027],'2028_ytd':[2028]}.items():
  q=x[x.index.year.isin(yrs)]; regs[n]={'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 and q.std(ddof=1)>0 else None,'hit_ratio':float((q>0).mean()) if len(q) else None}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
# Reconstruct all admitted signals for required pooled-cell Spearman evidence.
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def cb(x,y,cond=None,w=40):
 if cond is not None: x=x.where(cond);y=y.where(cond)
 return x.rolling(w,min_periods=12).cov(y)/y.rolling(w,min_periods=12).var()
def macro(n): return pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
dx=macro('DXY'); trend=(p/p.shift(20)-1)/v20; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/v20
orth=acc*np.nan
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
volpart=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v20[a] for a in A})
low=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A}); up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A})
spx=pd.DataFrame({a:-cb(r[a],r.SPX,w=20) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt(); aut=r.rolling(20,min_periods=15).corr(r.shift(1))
du=pd.DataFrame({a:-cb(r[a],dx,dx>0) for a in A});dd=pd.DataFrame({a:cb(r[a],dx,dx<0) for a in A}); asym=pd.DataFrame({a:cb(r[a],peer[a],peer[a]<0)-cb(r[a],peer[a],peer[a]>0) for a in A})
ua=pd.DataFrame({a:r[a].where(r[a].shift(1)>0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)>0))-r[a].where(r[a].shift(1)<0).rolling(40,min_periods=12).corr(r[a].shift(1).where(r[a].shift(1)<0)) for a in A})
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':low,'return_autocorrelation_20obs':aut,'relative_volume_participation_20d':volpart,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spx,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':up,'negative_conditional_dxy_up_beta_40obs':du,'positive_conditional_dxy_down_beta_40obs':dd,'asymmetric_peer_beta_resilience_40obs':asym,'upside_minus_downside_return_autocorr_40obs':ua}
print('FACTOR residual_volatility_expansion_10_60obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('COVERAGE',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]: print('METRIC',json.dumps(metrics(h),sort_keys=True))
mx=-1; who=''
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman');print('LIB',n,rho,len(z))
 if not np.isfinite(rho): mx=np.nan
 elif np.isfinite(mx) and abs(rho)>mx: mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if np.isfinite(mx) else 'MISSING_EVIDENCE')
