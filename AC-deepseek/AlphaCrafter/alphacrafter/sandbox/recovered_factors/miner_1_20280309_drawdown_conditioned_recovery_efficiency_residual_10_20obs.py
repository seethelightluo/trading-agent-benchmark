"""One candidate: residual drawdown-conditioned recovery path efficiency."""
import pandas as pd,numpy as np,json,glob,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-08'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); idx=p.index
v20=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/v20; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
# Recovery efficiency: recent 10d signed move per unit path length, activated continuously by prior 20d drawdown.
path=r.abs().rolling(10,min_periods=8).sum(); eff=(p/p.shift(10)-1)/path
dd=p/p.rolling(20,min_periods=15).max()-1
raw=eff*(-dd) # high only when an efficient rebound follows a material drawdown
# Cross-sectionally remove conventional trend and short reversal exposures each day.
f=raw*np.nan
for d in idx:
 z=pd.concat([raw.loc[d].rename('y'),trend.loc[d].rename('t'),rev.loc[d].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['t','r']]]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def metric(h):
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1); regs={}
 for n,yrs in {'2020_2022':[2020,2021,2022],'2023_2024':[2023,2024],'2025_2026':[2025,2026],'2027':[2027],'2028_ytd':[2028]}.items():
  y=x[x.index.year.isin(yrs)]; regs[n]={'dates':len(y),'ic':float(y.mean()) if len(y) else None,'icir':float(y.mean()/y.std(ddof=1)) if len(y)>1 and y.std(ddof=1)>0 else None,'hit':float((y>0).mean()) if len(y) else None}
 tos=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(tos)),'regimes':regs}
# Reconstruct every admitted signal; all correlations use finite pooled asset-date observations.
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def beta(x,y,w=40,cond=None,sign=1):
 if cond is None:cond=pd.Series(True,index=idx)
 return sign*x.where(cond).rolling(w,min_periods=12).cov(y.where(cond))/y.where(cond).rolling(w,min_periods=12).var()
def mret(name):
 d=pd.read_csv('../persistent/index_data/'+name+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END];return pd.to_numeric(d.close,errors='coerce').pct_change().reindex(idx)
dxy=mret('DXY');
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/v20; orth=acc*np.nan
for d in idx:
 z=pd.concat([acc.loc[d].rename('y'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];orth.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/v20[a] for a in A})
spxb=pd.DataFrame({a:beta(r[a],r.SPX,20,sign=-1) for a in A}); kurt=-r.rolling(40,min_periods=30).kurt(); autoc=r.rolling(20,min_periods=16).apply(lambda x:np.corrcoef(x[1:],x[:-1])[0,1] if np.std(x)>0 else np.nan,raw=True)
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});up=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A})
du=pd.DataFrame({a:beta(r[a],dxy,40,dxy>0,-1) for a in A}); ddB=pd.DataFrame({a:beta(r[a],dxy,40,dxy<0,1) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],40,peer[a]<0)-beta(r[a],peer[a],40,peer[a]>0) for a in A})
# volume factor with log only for strictly positive volume, so zero prints never create infinities.
part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce').where(lambda x:x>0)/pd.to_numeric(rd(a).volume,errors='coerce').where(lambda x:x>0).rolling(20,min_periods=1).mean()) for a in A})
updown=r.rolling(40,min_periods=30).apply(lambda x: np.corrcoef(x[x>0][1:],x[x>0][:-1])[0,1]-np.corrcoef(x[x<0][1:],x[x<0][:-1])[0,1] if (x>0).sum()>4 and (x<0).sum()>4 else np.nan,raw=True)
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':autoc,'relative_volume_participation_20d':part,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'upside_minus_downside_return_autocorr_40obs':updown,'inverse_expected_shortfall_40obs':es,'negative_spx_beta_20obs':spxb,'inverse_upside_peer_correlation_40obs':up,'negative_conditional_dxy_up_beta_40obs':du,'positive_conditional_dxy_down_beta_40obs':ddB,'asymmetric_peer_beta_resilience_40obs':asym,'inverse_excess_kurtosis_40obs':kurt}
print('FACTOR drawdown_conditioned_recovery_efficiency_residual_10_20obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H:print('METRIC',json.dumps(metric(h),sort_keys=True))
mx=0;who=''; ev={}
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.x,method='spearman')
 ev[n]={'rho':None if pd.isna(rho) else float(rho),'common_cells':len(z)};print('DIRECT_LIB',n,'rho',rho,'cells',len(z))
 if pd.isna(rho) or len(z)<8: raise RuntimeError('missing correlation evidence '+n)
 if abs(rho)>mx:mx=abs(rho);who=n
print('DIRECT_MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who);print('DIRECT_EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
