"""miner_1 single idea: volatility clustering 20 observations; validation through prior completed day."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-07-14')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={};v={}
for a in AS:p[a],v[a]=load(a)
def nf(fun,src=p):return pd.DataFrame({a:fun(x.dropna()).reindex(x.index) for a,x in src.items()})
r=nf(lambda x:x.pct_change()); med=r.median(axis=1); disp=r.std(axis=1)
def rs(x,y,w):
 out=[]
 for dt in x.index:
  z=pd.concat([x,y],axis=1).loc[:dt].tail(w).dropna();out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman') if len(z)>=max(15,w*3//4) else np.nan)
 return pd.Series(out,index=x.index)
# Candidate: serial persistence of absolute returns (high means clustered/turbulent volatility)
f=pd.DataFrame({a:r[a].abs().rolling(20,min_periods=15).apply(lambda z:pd.Series(z).autocorr(1),raw=False) for a in AS})
fw={h:pd.DataFrame({a:x.pct_change(h).shift(-h) for a,x in p.items()}) for h in (1,5,10,20)}
vol=nf(lambda x:x.pct_change().rolling(20,min_periods=15).std()); fast=nf(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std());slow=nf(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':nf(lambda x:np.log(x/x.rolling(20,min_periods=15).mean()),v),'volnorm_reversal_5obs':nf(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std()),'realized_volatility_20obs':vol,'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).apply(lambda z:z.autocorr(1),raw=False),'return_sign_balance_20obs':nf(lambda x:x.pct_change().gt(0).rolling(20,min_periods=15).mean()),'return_directional_efficiency_20obs':nf(lambda x:x.pct_change().rolling(20,min_periods=15).sum().abs()/x.pct_change().abs().rolling(20,min_periods=15).sum()),'relative_liquidity_stress_20_60obs':nf(lambda x:np.log(x.rolling(20,min_periods=15).mean()/x.rolling(60,min_periods=45).mean()),v),'dispersion_sensitivity_20obs':pd.DataFrame({a:rs(r[a],disp,20) for a in AS})}
ca={};rd={};vb={}
for a in AS:
 aa=[];dd=[];bb=[]
 for dt in r.index:
  z=pd.concat([r[a],med],axis=1).loc[:dt].tail(60).dropna();dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0]
  aa.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
  if len(z)>=45 and z.iloc[:,1].var()>0:
   beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var();e=z.iloc[:,0]-(z.iloc[:,0].mean()+beta*(z.iloc[:,1]-z.iloc[:,1].mean()));dd.append(np.sqrt(np.mean(np.minimum(e,0)**2))/np.sqrt(np.mean(e**2)));bb.append(beta)
  else:dd.append(np.nan);bb.append(np.nan)
 ca[a]=aa;rd[a]=dd;vb[a]=bb
lib['correlation_asymmetry_60obs']=pd.DataFrame(ca,index=r.index);lib['residual_downside_semivol_share_60obs']=pd.DataFrame(rd,index=r.index)
b=pd.DataFrame(vb,index=r.index);vv=nf(lambda x:x.pct_change().rolling(60,min_periods=45).std());lib['vol_orthogonal_median_beta_60obs']=b.sub(b.median(axis=1),axis=0)-vv.sub(vv.median(axis=1),axis=0).mul(b.corrwith(vv,axis=1).fillna(0),axis=0)
print('FACTOR volatility_clustering_autocorr_20obs: lag-1 autocorrelation of daily absolute returns over 20 observations');print('visible_through',END.date(),'assets',len(AS))
for h,y in fw.items():
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals));sd=x.std(ddof=1);print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[m];print(f' {n}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(tos):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.o,method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; library_count={len(lib)}')
