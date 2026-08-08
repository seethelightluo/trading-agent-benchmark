"""miner_1: downside relative resilience, a conditional cross-asset path-quality factor."""
import numpy as np,pandas as pd,glob,json
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-05-05')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={}; v={}
for a in AS:p[a],v[a]=load(a)
def nat(fn):return pd.DataFrame({a:fn(x.dropna()).reindex(x.index) for a,x in p.items()})
r=nat(lambda x:x.pct_change()); market=r.median(axis=1)
# Mean asset excess return on globally down sessions among its last 40 native observations; higher means resilience when cross-asset tape falls.
f=pd.DataFrame(index=r.index,columns=AS,dtype=float)
for a in AS:
 x=r[a]; vals=[]
 for dt in x.index:
  z=pd.concat([x,market],axis=1).loc[:dt].tail(40).dropna(); z=z[z.iloc[:,1]<0]
  vals.append((z.iloc[:,0]-z.iloc[:,1]).mean() if len(z)>=10 else np.nan)
 f[a]=pd.Series(vals,index=x.index)
fw={h:pd.DataFrame({a:(x.dropna().shift(-h)/x.dropna()-1) for a,x in p.items()}) for h in (1,5,10,20)}
vol=nat(lambda x:x.pct_change().rolling(20,min_periods=15).std()); fast=nat(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std());slow=nat(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':pd.DataFrame({a:np.log(x.dropna()/x.dropna().rolling(20,min_periods=15).mean()) for a,x in v.items()}),'volnorm_reversal_5obs':nat(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std()),'realized_volatility_20obs':vol,'trend_acceleration_20_60d':fast-slow,'correlation_asymmetry_60obs':pd.DataFrame(index=r.index,columns=AS),'return_skewness_20obs':r.rolling(20,min_periods=15).skew(),'return_persistence_autocorr_20obs':nat(lambda x:x.pct_change().rolling(20,min_periods=15).apply(lambda q:pd.Series(q).autocorr(),raw=False)),'return_sign_balance_20obs':nat(lambda x:(x.pct_change()>0).rolling(20,min_periods=15).mean()-.5),'return_directional_efficiency_20obs':nat(lambda x:(x/x.shift(20)-1).abs()/x.pct_change().abs().rolling(20,min_periods=15).sum())}
# beta compression and asymmetric correlation definitions
lib['beta_compression_20obs']=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(market) for a in AS})
for a in AS:
 vals=[]
 for dt in r.index:
  z=pd.concat([r[a],market],axis=1).loc[:dt].tail(60).dropna(); dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0]
  vals.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
 lib['correlation_asymmetry_60obs'][a]=vals
# active files exclude deprecated, but reproduce 10 known effective signals; output all correlations as evidence
print('FACTOR downside_relative_resilience_40obs = mean(r_asset-r_crossasset_median | median return <0) over trailing 40 native observations; require 10 down observations')
print('visible through',END.date(),'universe',len(AS))
allics={}
for h,y in fw.items():
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 q=pd.Series(dict(vals));allics[h]=q; sd=q.std(ddof=1)
 print(f'H={h} dates={len(q)} meanIC={q.mean():.6f} ICIR={q.mean()/sd:.6f} hit={(q>0).mean():.4f} coverage={np.mean(cov):.4f}')
 if h==20:
  for n,mask in [('2020',q.index<'2021-01-01'),('2021-22',(q.index>='2021-01-01')&(q.index<'2023-01-01')),('2023-24',(q.index>='2023-01-01')&(q.index<'2025-01-01')),('2025-current',q.index>='2025-01-01')]:
   w=q[mask];print(f' {n}: n={len(w)} IC={w.mean():.6f} ICIR={w.mean()/w.std(ddof=1):.6f} hit={(w>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turn):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1;who=''; cells=0
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).dropna(); rho=z.f.corr(z.o,method='spearman');cells=max(cells,len(z));print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);who=n
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; library_correlation_evidence_cells={cells}; library_json_count={len(glob.glob("factors/*.json"))}')
