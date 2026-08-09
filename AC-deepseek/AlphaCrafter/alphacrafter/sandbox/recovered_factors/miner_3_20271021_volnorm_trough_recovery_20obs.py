"""One-factor validation: volatility-normalized recovery from 20-day trough."""
import pandas as pd,numpy as np,json,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-20'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A});r=p.pct_change();vol=r.rolling(20,min_periods=15).std()
# Current rebound from trailing 20-day low, normalized by contemporaneous 20d realized volatility.
f=(p/p.rolling(20,min_periods=15).min()-1)/vol
print('FACTOR volnorm_trough_recovery_20obs = (close / rolling_min(close,20) - 1) / rolling_std(return,20); recent recovery strength normalized for asset volatility')
print('visible_through',END.date(),'data_range',p.index.min().date(),p.index.max().date(),'assets',len(A),'library_files',len(glob.glob('factors/*.json')))
def met(h):
 fw=p.shift(-h)/p-1;q=[];nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 x=pd.Series(dict(q));sd=x.std(ddof=1);reg={}
 for name,mask in [('2020',x.index.year==2020),('2021_2022',x.index.year.isin([2021,2022])),('2023_2024',x.index.year.isin([2023,2024])),('2025_2026',x.index.year.isin([2025,2026])),('2027',x.index.year==2027)]:
  y=x[mask];reg[name]={'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit':float((y>0).mean())}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic_dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(nn)),'turnover_10d':float(np.mean(turn)),'regimes':reg}
print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H:print('METRIC',json.dumps(met(h),sort_keys=True))
# admitted library reconstructions
trend=(p/p.shift(20)-1)/vol;rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=r.index,columns=A,dtype=float)
for dt in r.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:orth.loc[dt,z.index]=z.a-np.c_[np.ones(len(z)),z.t]@np.linalg.lstsq(np.c_[np.ones(len(z)),z.t],z.a,rcond=None)[0]
spx=r.SPX;spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(spx)/spx.rolling(20,min_periods=15).var() for a in A})
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(r.index);dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A});part=pd.DataFrame({a:np.log(rd(a).volume.astype(float)/rd(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt();es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(r.index);va=pd.DataFrame(index=r.index,columns=A);du=va.copy();dd=va.copy()
for a in A:
 for i in range(39,len(r)):
  x=r[a].iloc[i-39:i+1];y=dxy.iloc[i-39:i+1]
  for m,o,sgn in [(y>0,du,-1),(y<0,dd,1)]:
   if m.sum()>=12 and y[m].var()>0:o.iloc[i,o.columns.get_loc(a)]=sgn*x[m].cov(y[m])/y[m].var()
 for i in range(59,len(r)):
  x=r[a].iloc[i-59:i+1];y=vix.iloc[i-59:i+1];u=y>0;d=y<0
  if u.sum()>=10 and d.sum()>=10:va.iloc[i,va.columns.get_loc(a)]=x[u].cov(y[u])/y[u].var()-x[d].cov(y[d])/y[d].var()
low=pd.DataFrame(index=r.index,columns=A);hi=low.copy()
for a in A:
 y=r.drop(columns=a).mean(axis=1)
 for i in range(39,len(r)):
  x=r[a].iloc[i-39:i+1];q=y.iloc[i-39:i+1]
  for m,o,sgn in [(q<0,low,1),(q>0,hi,-1)]:
   if m.sum()>=12:o.iloc[i,o.columns.get_loc(a)]=sgn*x[m].corr(q[m])
lib={'risk_adjusted_trend':trend,'volnorm_reversal':rev,'inverse_realized_volatility':-vol,'relative_volume':part,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asymmetry':va,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':low,'inverse_upside_peer_correlation':hi,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd}
mx=-1
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('LIB',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
