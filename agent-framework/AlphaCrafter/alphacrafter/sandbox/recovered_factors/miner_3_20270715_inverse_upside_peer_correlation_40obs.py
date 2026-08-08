"""miner_3: conditional upside peer-correlation factor, 40 observations.
At each date, signal is negative correlation of asset return with equal-weighted
peer return, measured only on days when the peer return was positive. It tests
whether assets retaining independent behavior during broad advances predict
subsequent cross-asset return. No data after END are used for signal formation.
"""
import pandas as pd, numpy as np, glob, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-07-14'); H=[1,5,10,20]
P={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float)
p=pd.DataFrame(P); r=p.pct_change(); fwd={h:p.shift(-h)/p-1 for h in H}
# signal: - conditional correlation, 40 calendar observations with >=12 positive-peer observations
sig=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 peer=r.drop(columns=a).mean(axis=1)
 for k in range(39,len(r)):
  x=r[a].iloc[k-39:k+1]; y=peer.iloc[k-39:k+1]; m=(y>0)&x.notna()&y.notna()
  if m.sum()>=12 and x[m].std()>0 and y[m].std()>0: sig.iloc[k,sig.columns.get_loc(a)]=-x[m].corr(y[m])
print('FACTOR inverse_upside_peer_correlation_40obs = -corr_40(r_asset, mean(r_peers) | mean(r_peers)>0); min conditional observations=12')
print('asof',END.date(),'instruments',len(A),'library_files',len([x for x in glob.glob('factors/*.json') if not x.endswith('.bak')]))
ics={}
for h in H:
 vals=[]; cov=[]; ni=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt].rename('f'),fwd[h].loc[dt].rename('r')],axis=1).dropna();cov.append(len(z)/15)
  if len(z)>=8: vals.append((dt,z.f.corr(z.r,method='spearman')));ni.append(len(z))
 x=pd.Series(dict(vals)); ics[h]=x
 print(f'h={h} IC_dates={len(x)} IC={x.mean():+.6f} ICIR={x.mean()/x.std(ddof=1):+.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f} mean_instruments={np.mean(ni):.2f}')
for label,lo,hi in [('2020','2020-01-01','2021-01-01'),('2021_2022','2021-01-01','2023-01-01'),('2023_2024','2023-01-01','2025-01-01'),('2025_2026','2025-01-01','2027-01-01'),('2027','2027-01-01','2100-01-01')]:
 x=ics[10][(ics[10].index>=lo)&(ics[10].index<hi)]
 print(f'regime_{label}_h10 n={len(x)} IC={x.mean():+.6f} ICIR={x.mean()/x.std(ddof=1):+.6f} hit={(x>0).mean():.4f}')
rk=sig.rank(axis=1,pct=True); ts=[]
for k in range(10,len(rk),10):
 z=pd.concat([rk.iloc[k-10],rk.iloc[k]],axis=1).dropna()
 if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_10obs={np.mean(ts):.6f}; factor_cells={sig.notna().sum().sum()}/{sig.size} ({sig.notna().mean().mean():.4f})')
# Reconstruct each admitted definition sufficiently for binding correlation evidence.
def macro(n): return pd.read_csv(f'../persistent/index_data/{n}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol
rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=pd.DataFrame(index=r.index,columns=A,dtype=float)
for dt in r.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  b=np.polyfit(z.t,z.a,1); orth.loc[dt,z.index]=z.a-b[0]*z.t-b[1]
spx=r.SPX; dxy=macro('DXY'); vix=macro('VIX')
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(spx)/spx.rolling(20,min_periods=15).var() for a in A})
dxyb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A})
# asymmetric VIX beta: conditional + minus conditional - rolling slopes
vixa=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 for k in range(59,len(r)):
  x=r[a].iloc[k-59:k+1];y=vix.reindex(r.index).iloc[k-59:k+1]; pos=y>0; neg=y<0
  if pos.sum()>=10 and neg.sum()>=10 and y[pos].var()>0 and y[neg].var()>0:vixa.iloc[k,vixa.columns.get_loc(a)]=x[pos].cov(y[pos])/y[pos].var()-x[neg].cov(y[neg])/y[neg].var()
kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/r[a].rolling(20,min_periods=15).std() for a in A})
# current admitted downside peer correlation exact symmetrical construction
low=pd.DataFrame(index=r.index,columns=A,dtype=float)
for a in A:
 y=r.drop(columns=a).mean(axis=1)
 for k in range(39,len(r)):
  x=r[a].iloc[k-39:k+1];q=y.iloc[k-39:k+1];m=q<0
  if m.sum()>=12 and x[m].std()>0 and q[m].std()>0:low.iloc[k,low.columns.get_loc(a)]=x[m].corr(q[m])
# participation (if volumes unavailable for some asset, NaNs are retained)
parts={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 parts[a]=np.log(d.volume.astype(float)/d.volume.astype(float).rolling(20,min_periods=1).mean()) if 'volume' in d else pd.Series(index=r.index,dtype=float)
part=pd.DataFrame(parts).reindex(r.index)
lib={'risk_adjusted_trend':trend,'ravmom_duplicate':trend,'volnorm_reversal':rev,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'dxy_beta':dxyb,'vix_asym_beta':vixa,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':low,'relative_volume_participation':part}
mx=-1
for n,s in lib.items():
 z=pd.concat([sig.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman');print(f'library_{n}_rho={rho:+.6f}; common_cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);who=n
print(f'max_abs_library_correlation={mx:.6f}; factor={who}')
