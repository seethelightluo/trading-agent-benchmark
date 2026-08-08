"""One idea: regime-switched relative resilience: trend in high dispersion, short reversal in low dispersion."""
import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2028-01-12')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index);return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A});idx=p.index;r=p.pct_change();vol20=r.rolling(20,min_periods=15).std().replace(0,np.nan)
# Cross-asset dispersion is an observable market-state variable. High dispersion rewards persistent relative resilience; low dispersion favors mean reversion.
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); high=disp>disp.rolling(60,min_periods=40).median()
trend=(p/p.shift(20)-1)/vol20; reversal=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std().replace(0,np.nan)
f=trend.where(high, reversal)
def met(h):
 fw=p.shift(-h)/p-1;out=[];nn=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));nn.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1); regs={}
 for nm,yrs in {'2020_2022':[2020,2021,2022],'2023_2024':[2023,2024],'2025_2026':[2025,2026],'2027':[2027],'2028':[2028]}.items():
  q=x[x.index.year.isin(yrs)];regs[nm]={'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 and q.std(ddof=1)>0 else None,'hit':float((q>0).mean()) if len(q) else None}
 to=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(nn)),'turnover_10d':float(np.mean(to)),'regimes':regs}
print('FACTOR dispersion_conditional_relative_resilience_20obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.replace([np.inf,-np.inf],np.nan).count().sum()),'/',f.size,float(f.replace([np.inf,-np.inf],np.nan).count().sum()/f.size),'high_dispersion_share',float(high.mean()))
for h in [1,5,10,20]: print('METRIC',json.dumps(met(h),sort_keys=True))
# Full finite pooled comparisons against current admitted library; signals reconstructed from their published definitions.
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def roll(x,y,w,sel,kind='beta',sign=1,mn=12):
 o=[]
 for i in range(len(x)):
  xx=x.iloc[max(0,i-w+1):i+1].to_numpy(float);yy=y.iloc[max(0,i-w+1):i+1].to_numpy(float);q=sel(yy)&np.isfinite(xx)&np.isfinite(yy);xx=xx[q];yy=yy[q]
  o.append(sign*(np.corrcoef(xx,yy)[0,1] if kind=='corr' and len(xx)>=mn and np.std(xx)>0 and np.std(yy)>0 else (np.cov(xx,yy,ddof=1)[0,1]/np.var(yy,ddof=1) if len(xx)>=mn and np.var(yy,ddof=1)>0 else np.nan)))
 return pd.Series(o,index=idx)
def betad(y,w=40,sel=lambda z:np.ones(len(z),bool),sign=1):return pd.DataFrame({a:roll(r[a],y,w,sel,'beta',sign) for a in A})
def corrd(sel,sign=1):return pd.DataFrame({a:roll(r[a],peer[a],40,sel,'corr',sign) for a in A})
acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol20;orth=acc*np.nan
for d in idx:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8: orth.loc[d,z.index]=z.iloc[:,0]-np.c_[np.ones(len(z)),z.iloc[:,1]]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.iloc[:,1]],z.iloc[:,0],rcond=None)[0]
kurt=-r.rolling(40,min_periods=30).kurt();es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.nanquantile(x,.2)]),raw=True)/vol20[a] for a in A})
autoc=r.rolling(20,min_periods=15).corr(r.shift(1));spxb=betad(r['SPX'],20,sign=-1)
dxydata=get_index_daily_data('DXY',5000).set_index('date');dxydata.index=pd.to_datetime(dxydata.index);dxy=pd.to_numeric(dxydata.loc[:END,'close'],errors='coerce').pct_change().reindex(idx);dxyb=betad(dxy,20);du=betad(dxy,40,lambda z:z>0,-1);dd=betad(dxy,40,lambda z:z<0)
vdata=get_index_daily_data('VIX',5000).set_index('date');vdata.index=pd.to_datetime(vdata.index);vix=pd.to_numeric(vdata.loc[:END,'close'],errors='coerce').pct_change().reindex(idx);va=betad(vix,60,lambda z:z>0)-betad(vix,60,lambda z:z<0)
part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce').replace(0,np.nan)/pd.to_numeric(rd(a).volume,errors='coerce').replace(0,np.nan).rolling(20,min_periods=5).mean()) for a in A}).replace([np.inf,-np.inf],np.nan)
down=corrd(lambda z:z<0);up=corrd(lambda z:z>0,-1);asym=pd.DataFrame({a:roll(r[a],peer[a],40,lambda z:z<0)-roll(r[a],peer[a],40,lambda z:z>0) for a in A});asympeer=pd.DataFrame({a:roll(r[a],peer[a],40,lambda z:z<0)-roll(r[a],peer[a],40,lambda z:z>0,sign=-1) for a in A})
# conditional own-return autocorrelation and published upside-minus-downside variant
ownup=pd.DataFrame({a:roll(r[a],r[a].shift(1),40,lambda z:z>0,'corr') for a in A});owndn=pd.DataFrame({a:roll(r[a],r[a].shift(1),40,lambda z:z<0,'corr') for a in A});ownas=ownup-owndn
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':reversal,'downside_peer_correlation_40obs':down,'return_autocorrelation_20obs':autoc,'relative_volume_participation_20d':part,'risk_adjusted_trend_20d':trend,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':spxb,'dxy_beta_20obs':dxyb,'vix_asymmetric_shock_beta_60obs':va,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':es,'inverse_upside_peer_correlation_40obs':up,'negative_conditional_dxy_up_beta_40obs':du,'positive_conditional_dxy_down_beta_40obs':dd,'asymmetric_peer_beta_resilience_40obs':asympeer,'upside_minus_downside_return_autocorr_40obs':ownas}
mx=0;who='';bad=[]
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan;print('LIB',n,'rho',rho,'cells',len(z))
 if not np.isfinite(rho):bad.append(n)
 else:
  if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',None if bad else mx,'FACTOR',who,'MISSING',bad)
"""
