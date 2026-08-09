"""Miner_3: validate overnight-loss intraday-recovery resilience, distinct from close location."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-20')
def field(a,col='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=END,col],errors='coerce')
P=pd.DataFrame({a:field(a) for a in A}); O=pd.DataFrame({a:field(a,'open') for a in A}); H=pd.DataFrame({a:field(a,'high') for a in A}); Lo=pd.DataFrame({a:field(a,'low') for a in A}); R=P.pct_change(fill_method=None)
def csres(x,y):
 q=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-15:
   b=z.x.cov(z.y)/z.y.var();q.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return q
v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std(); v60=R.rolling(60,min_periods=45).std(); mom=(P/P.shift(20)-1)/(v20+1e-12)
# Mean normalized same-session recovery after an adverse opening. This differs from location: it measures return magnitude from open to close, conditional on an overnight loss.
gap=O/P.shift(1)-1; intra=P/O-1; recovery=(intra/(v20+1e-12)).where(gap<0).rolling(20,min_periods=6).mean(); F=csres(recovery,mom)
L={'ravmom':mom,'reversal5':-(P/P.shift(5)-1)/(v5+1e-12),'realized_vol':v20}
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A});L['peer_crowding']=peer
def macro(x):
 d=pd.read_csv('../persistent/index_data/'+x+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=END,'close'],errors='coerce').reindex(P.index)
for n,k in [('VIX','vix_resid'),('DXY','dxy_resid')]:L[k]=csres(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(macro(n).pct_change(fill_method=None)) for a in A}),peer)
vix=macro('VIX');L['vix_peer']=peer.mul((vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
L['low_vov']=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()); market=R.mean(axis=1)
bd=pd.DataFrame(index=P.index,columns=A);bu=bd.copy()
for a in A:
 bd[a]=R[a].where(market<0).rolling(30,min_periods=6).cov(market.where(market<0))/market.where(market<0).rolling(30,min_periods=6).var();bu[a]=R[a].where(market>0).rolling(30,min_periods=6).cov(market.where(market>0))/market.where(market>0).rolling(30,min_periods=6).var()
L['downside_beta']=-(bd-bu); pos=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);L['inverse_upside_resid']=-csres(pos,mom)
I=(R<0).astype(float);L['loss_clustering']=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
L['inverse_skew']=csres(csres(-R.rolling(20,min_periods=15).skew(),mom),v20)
loc=(P-Lo)/(H-Lo).replace(0,np.nan); L['close_location']=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
dd=P/P.rolling(20,min_periods=15).max()-1;L['drawdown_inverse_ac']=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A})
print('FACTOR overnight_loss_intraday_recovery_residual_20; visible_through',END.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; x=[]; ns=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:x.append((t,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(x));ics[h]=x; sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for lab,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',ics[10].index>='2026-01-01')]:
 x=ics[10][mask];print('regime',lab,'dates',len(x),'IC',None if len(x)==0 else round(x.mean(),6),'ICIR',None if len(x)<2 else round(x.mean()/x.std(ddof=1),6),'hit',None if len(x)==0 else round((x>0).mean(),4))
r=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(ts):.6f}')
mx=-1
for k,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',k,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=k;nc=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',nc,'records',len(glob.glob('factors/*.json')))
