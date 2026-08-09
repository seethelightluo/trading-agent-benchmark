"""One idea: 20-observation close-location pressure, de-meaned by own 60d norm."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-01-27'); P={}; H={}; LO={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index()
 P[a]=pd.to_numeric(d.close,errors='coerce');H[a]=pd.to_numeric(d.high,errors='coerce');LO[a]=pd.to_numeric(d.low,errors='coerce')
P=pd.DataFrame(P);H=pd.DataFrame(H);LO=pd.DataFrame(LO);R=P.pct_change(fill_method=None)
# A persistently high close within each daily range denotes buying pressure; subtracting
# the instrument's own 60d baseline makes this a local flow/pressure surprise.
loc=(P-LO)/(H-LO).replace(0,np.nan)
F=loc.rolling(20,min_periods=15).mean()-loc.rolling(60,min_periods=40).mean()
# admitted library reconstructions
L={};vol=R.rolling(20,min_periods=15).std();L['ravmom']=(P/P.shift(20)-1)/vol;L['volnorm_reversal']=-(P/P.shift(5)-1)/R.rolling(5,min_periods=4).std();L['realized_vol']=vol
peer=pd.DataFrame(index=P.index,columns=A)
for a in A: peer[a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1)
L['peer_crowding']=peer
def resid(x,y):
 out=pd.DataFrame(index=x.index,columns=A)
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>0:
   b=np.cov(z.x,z.y,ddof=1)[0,1]/z.y.var();out.loc[dt,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return out
def macro(fn):
 m=pd.read_csv('../persistent/index_data/'+fn,parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').close.astype(float).pct_change(fill_method=None).reindex(P.index)
 return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),peer)
L['vix_beta_resid']=macro('VIX.csv');L['dxy_beta_resid']=macro('DXY.csv')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index)
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5); asym=np.log((up+1e-8)/(dn+1e-8)); L['high_vix_asym']=resid(asym,P/P.shift(20)-1).mul((v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
print('FACTOR close_location_pressure_surprise_20; visible through',END.date(),'assets',len(A));print('expression=mean20((close-low)/(high-low))-mean60((close-low)/(high-low))')
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[];nn=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((dt,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 x=pd.Series(dict(vals));print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(nn):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
 if h==5:
  for n,q in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
   y=x[q];print(n,'dates',len(y),f'IC={y.mean():.6f}',f'ICIR={y.mean()/y.std(ddof=1):.6f}')
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',np.mean(turn),'signal_cell_coverage',F.notna().mean().mean())
mx=0
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');print('library',n,'rho',q,'cells',len(z));mx=max(mx,abs(q))
print('max_abs_library_correlation',mx,'library_records',len([x for x in glob.glob('factors/*.json') if not x.endswith('.bak')]))
