"""One candidate: inverse volatility-ratio (10d / 60d) compression."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-09'); H=[1,5,10,20]
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A}); r=p.pct_change(); idx=p.index
v10=r.rolling(10,min_periods=8).std(); v20=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=45).std()
# Higher score means recent volatility is compressed relative to the asset's own medium-run volatility.
f=-(v10/v60)
def metric(h):
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for d in idx:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1); regs={}
 for n,mask in [('2020_2022',x.index.year<=2022),('2023_2024',x.index.year.isin([2023,2024])),('2025_2026',x.index.year.isin([2025,2026])),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028)]:
  y=x[mask];regs[n]={'dates':len(y),'ic':float(y.mean()) if len(y) else None,'icir':float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit':float((y>0).mean()) if len(y) else None}
 tos=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(tos)),'regimes':regs}
def cond(x,y,w,sel,sgn=1):
 o=[]
 for i in range(len(x)):
  xx=x.iloc[max(0,i-w+1):i+1];yy=y.iloc[max(0,i-w+1):i+1];q=sel(yy)&xx.notna()&yy.notna()
  o.append(sgn*np.cov(xx[q],yy[q],ddof=1)[0,1]/np.var(yy[q],ddof=1) if q.sum()>=10 and np.var(yy[q],ddof=1)>0 else np.nan)
 return pd.Series(o,index=idx)
def peer(kind):
 return pd.DataFrame({a:cond(r[a],r.drop(columns=a).mean(1),40,(lambda x:x<0) if kind=='down' else lambda x:x>0,1 if kind=='down' else -1) for a in A})
def macro(x):return pd.read_csv('../persistent/index_data/'+x+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(idx)
trend=(p/p.shift(20)-1)/v20; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); autoc=r.rolling(20,min_periods=16).apply(lambda x:np.corrcoef(x[1:],x[:-1])[0,1] if np.std(x)>0 else np.nan,raw=True)
dxy=macro('DXY'); spx=-pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A}); low=peer('down');up=peer('up')
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'downside_peer_correlation_40obs':low,'return_autocorrelation_20obs':autoc,'inverse_realized_volatility_20obs':-v20,'negative_spx_beta_20obs':spx,'dxy_beta_20obs':pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(dxy)/dxy.rolling(20,min_periods=15).var() for a in A}),'inverse_upside_peer_correlation_40obs':up}
print('FACTOR inverse_volatility_compression_10_60obs visible_through',END.date(),'period',idx.min().date(),idx.max().date(),'assets',len(A));print('coverage',int(f.notna().sum().sum()),'/',f.size,float(f.notna().mean().mean()))
for h in H:print('METRIC',json.dumps(metric(h),sort_keys=True))
mx=-1;who='';ev={}
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');ev[n]={'rho':None if pd.isna(rho) else float(rho),'common_cells':len(z)}
 print('DIRECT_LIB',n,'rho',rho,'cells',len(z))
 if pd.isna(rho):mx=np.nan
 elif not pd.isna(mx) and abs(rho)>mx:mx=abs(rho);who=n
print('DIRECT_MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who if not pd.isna(mx) else 'MISSING_EVIDENCE');print('DIRECT_EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
