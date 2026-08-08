"""miner_3 candidate: negative conditional DXY-up beta, 40 observations."""
import pandas as pd,numpy as np,glob,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-28'); H=[1,5,10,20]
def read(a): return pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:read(a).close.astype(float) for a in A});r=p.pct_change(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(r.index); fwd={h:p.shift(-h)/p-1 for h in H}
sig=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 for k in range(39,len(r)):
  x=r[a].iloc[k-39:k+1]; y=dxy.iloc[k-39:k+1]; m=(y>0)&x.notna()&y.notna()
  if m.sum()>=12 and y[m].var()>0: sig.iloc[k,sig.columns.get_loc(a)]=-x[m].cov(y[m])/y[m].var()
print('FACTOR negative_conditional_dxy_up_beta_40obs = -beta_40(r_asset, r_DXY | r_DXY>0); min conditional observations=12')
print('asof',END.date(),'instruments',len(A),'library_files',len(glob.glob('factors/*.json')))
ics={}
for h in H:
 vals=[];cv=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt].rename('f'),fwd[h].loc[dt].rename('r')],axis=1).dropna();cv.append(len(z)/15)
  if len(z)>=8: vals.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x
 print(f'h={h} IC_dates={len(x)} IC={x.mean():+.6f} ICIR={x.mean()/x.std(ddof=1):+.6f} hit={(x>0).mean():.4f} coverage={np.mean(cv):.4f} mean_instruments={np.mean(ns):.2f}')
for lab,lo,hi in [('2020','2020','2021'),('2021_2022','2021','2023'),('2023_2024','2023','2025'),('2025_2026','2025','2027'),('2027','2027','2100')]:
 x=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)];print(f'regime_{lab}_h10 n={len(x)} IC={x.mean():+.6f} ICIR={x.mean()/x.std(ddof=1):+.6f} hit={(x>0).mean():.4f}')
rk=sig.rank(axis=1,pct=True);to=[]
for k in range(10,len(rk),10):
 z=pd.concat([rk.iloc[k-10],rk.iloc[k]],axis=1).dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_10obs={np.mean(to):.6f}; factor_cells={sig.notna().sum().sum()}/{sig.size} ({sig.notna().mean().mean():.4f})')
# admitted-factor correlation evidence (definitions match research library)
vol=r.rolling(20,min_periods=15).std();trend=(p/p.shift(20)-1)/vol;rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=r.index,columns=A,dtype=float)
for dt in r.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  b=np.polyfit(z.t,z.a,1);orth.loc[dt,z.index]=z.a-b[0]*z.t-b[1]
spx=r.SPX;spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(spx)/spx.rolling(20,min_periods=15).var() for a in A});dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A})
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(r.index);vixa=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 for k in range(59,len(r)):
  x=r[a].iloc[k-59:k+1];y=vix.iloc[k-59:k+1];po=y>0;ne=y<0
  if po.sum()>=10 and ne.sum()>=10 and y[po].var()>0 and y[ne].var()>0:vixa.iloc[k,vixa.columns.get_loc(a)]=x[po].cov(y[po])/y[po].var()-x[ne].cov(y[ne])/y[ne].var()
kurt=-r.rolling(40,min_periods=30).kurt();es=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/r[a].rolling(20,min_periods=15).std() for a in A});low=pd.DataFrame(index=r.index,columns=A,dtype=float);up=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 y=r.drop(columns=a).mean(axis=1)
 for k in range(39,len(r)):
  x=r[a].iloc[k-39:k+1];q=y.iloc[k-39:k+1]
  for m,out,sgn in [(q<0,low,1),(q>0,up,-1)]:
   if m.sum()>=12 and x[m].std()>0 and q[m].std()>0:out.iloc[k,out.columns.get_loc(a)]=sgn*x[m].corr(q[m])
part=pd.DataFrame({a:np.log(read(a).volume.astype(float)/read(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A}).reindex(r.index)
lib={'risk_adjusted_trend':trend,'ravmom':trend,'volnorm_reversal':rev,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asym_beta':vixa,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':low,'inverse_upside_peer_correlation':up,'relative_volume_participation':part};mx=-1
for n,s in lib.items():
 z=pd.concat([sig.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman');print(f'library_{n}_rho={rho:+.6f}; common_cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);who=n
print(f'max_abs_library_correlation={mx:.6f}; factor={who}')
