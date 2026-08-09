"""miner_1 single idea: 60d cross-asset residual-return skewness."""
import numpy as np,pandas as pd
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2027-06-16')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 return d.close.astype(float),d.volume.astype(float).replace(0,np.nan)
p={};v={}
for a in AS:p[a],v[a]=load(a)
def nf(fun,src=p):return pd.DataFrame({a:fun(x.dropna()).reindex(x.index) for a,x in src.items()})
r=nf(lambda x:x.pct_change());med=r.median(axis=1)
# Rolling residual distribution after removal of contemporaneous cross-asset median return.
def rsk(a):
 out=[]
 for dt in r.index:
  z=pd.concat([r[a],med],axis=1).loc[:dt].tail(60).dropna()
  if len(z)<45 or z.iloc[:,1].var()==0:out.append(np.nan);continue
  b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); e=z.iloc[:,0]-(z.iloc[:,0].mean()+b*(z.iloc[:,1]-z.iloc[:,1].mean()))
  out.append(e.skew())
 return pd.Series(out,index=r.index)
f=pd.DataFrame({a:rsk(a) for a in AS})
fw={h:pd.DataFrame({a:x.pct_change(h).shift(-h) for a,x in p.items()}) for h in(1,5,10,20)}
vol=nf(lambda x:x.pct_change().rolling(20,min_periods=15).std());fast=nf(lambda x:(x/x.shift(20)-1)/x.pct_change().rolling(20,min_periods=15).std());slow=nf(lambda x:(x/x.shift(60)-1)/x.pct_change().rolling(60,min_periods=45).std())
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':nf(lambda x:np.log(x/x.rolling(20,min_periods=15).mean()),v),'volnorm_reversal_5obs':nf(lambda x:-(x/x.shift(5)-1)/x.pct_change().rolling(5,min_periods=4).std()),'realized_volatility_20obs':vol,'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).apply(lambda z:z.autocorr(1),raw=False),'return_sign_balance_20obs':nf(lambda x:x.pct_change().gt(0).rolling(20,min_periods=15).mean()),'return_directional_efficiency_20obs':nf(lambda x:x.pct_change().rolling(20,min_periods=15).sum().abs()/x.pct_change().abs().rolling(20,min_periods=15).sum()),'relative_liquidity_stress_20_60obs':nf(lambda x:np.log(x.rolling(20,min_periods=15).mean()/x.rolling(60,min_periods=45).mean()),v)}
asym={};resid={}
for a in AS:
 aa=[];rr=[]
 for dt in r.index:
  z=pd.concat([r[a],med],axis=1).loc[:dt].tail(60).dropna();dn=z[z.iloc[:,1]<0];up=z[z.iloc[:,1]>=0]
  aa.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
  if len(z)>=60 and z.iloc[:,1].var()>0:
   b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var();e=z.iloc[:,0]-(z.iloc[:,0].mean()+b*(z.iloc[:,1]-z.iloc[:,1].mean()));rr.append(np.sqrt(np.mean(np.minimum(e,0)**2))/np.sqrt(np.mean(e**2)))
  else:rr.append(np.nan)
 asym[a]=aa;resid[a]=rr
lib['correlation_asymmetry_60obs']=pd.DataFrame(asym,index=r.index);lib['residual_downside_semivol_share_60obs']=pd.DataFrame(resid,index=r.index)
print('FACTOR residual_return_skewness_60obs: skewness of 60 daily residual returns after rolling beta removal against cross-asset median return')
print('visible_through',END.date(),'assets',len(AS))
for h,y in fw.items():
 vals=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:vals.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals));sd=x.std(ddof=1);print(f'H={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 for nm,ma in [('2020',x.index<'2021-01-01'),('2021-22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023-24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025-current',x.index>='2025-01-01')]:
  q=x[ma];print(f' {nm}: n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f}; signal_cells={f.notna().sum().sum()}/{f.size}={f.notna().mean().mean():.4f}')
mx=-1
for n,o in lib.items():
 z=pd.concat([f.stack().rename('f'),o.stack().rename('o')],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=z.f.corr(z.o,method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);closest=n
print(f'max_abs_library_correlation={mx:.6f}; closest={closest}; library_count={len(lib)}')
