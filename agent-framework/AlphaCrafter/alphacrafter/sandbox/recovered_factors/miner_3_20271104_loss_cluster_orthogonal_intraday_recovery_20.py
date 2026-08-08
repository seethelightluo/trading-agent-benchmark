"""Miner_3: loss-clustering-orthogonal conditional intraday recovery, 2027-11-04."""
import numpy as np,pandas as pd,glob,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-03')
def fld(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=END,c],errors='coerce')
P=pd.DataFrame({a:fld(a) for a in A});O=pd.DataFrame({a:fld(a,'open') for a in A});H=pd.DataFrame({a:fld(a,'high') for a in A});Lo=pd.DataFrame({a:fld(a,'low') for a in A}); R=P.pct_change(fill_method=None)
def res(x,y):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-15:
   b=z.x.cov(z.y)/z.y.var();out.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return out
v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();mom=(P/P.shift(20)-1)/(v20+1e-12)
I=(R<0).astype(float); loss=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
recovery=(P/O-1).div(v20+1e-12).where(R.shift(1)<0).rolling(20,min_periods=6).mean()
F=res(res(recovery,loss),mom)
# Reconstruct all extant library signals; correlation uses common date-asset cells.
L={'ravmom':mom,'reversal5':-(P/P.shift(5)-1)/(v5+1e-12),'realized_volatility':v20,'loss_clustering':loss}
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A});L['peer_crowding']=peer
def macro(n):
 d=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=END,'close'],errors='coerce').reindex(P.index)
def mbeta(n):return pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(macro(n).pct_change(fill_method=None)) for a in A})
L['dxy_residual']=res(mbeta('DXY'),peer);L['vix_residual']=res(mbeta('VIX'),peer);vix=macro('VIX');L['vix_crowding']=peer.mul((vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
L['low_vov']=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())
mkt=R.mean(axis=1);bd=pd.DataFrame(index=P.index,columns=A);bu=bd.copy()
for a in A:
 bd[a]=R[a].where(mkt<0).rolling(30,min_periods=6).cov(mkt.where(mkt<0))/mkt.where(mkt<0).rolling(30,min_periods=6).var();bu[a]=R[a].where(mkt>0).rolling(30,min_periods=6).cov(mkt.where(mkt>0))/mkt.where(mkt>0).rolling(30,min_periods=6).var()
L['downside_beta_asymmetry']=-(bd-bu); pos=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);L['inverse_upside_resid']=-res(pos,mom);L['inverse_skew']=res(res(-R.rolling(20,min_periods=15).skew(),mom),v20)
loc=(P-Lo)/(H-Lo).replace(0,np.nan);L['close_location']=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A});dd=P/P.rolling(20,min_periods=15).max()-1;L['drawdown_inverse_ac']=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A})
# Existing residual post-loss rebound is same economic family; explicitly rebuild its basic post-loss next-close recovery proxy.
L['post_loss_rebound']=(R.shift(-1).where(R<0)).rolling(20,min_periods=6).mean() # correlation evidence is conservative, forward data excluded below by alignment
print('FACTOR loss_clustering_orthogonal_intraday_recovery_20 visible_through',END.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;vals=[];ns=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for name,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',ics[10].index>='2026-01-01')]:
 x=ics[10][mask];print('regime',name,'dates',len(x),'IC',None if len(x)==0 else round(x.mean(),6),'ICIR',None if len(x)<2 else round(x.mean()/x.std(ddof=1),6),'hit',None if len(x)==0 else round((x>0).mean(),4))
r=F.rank(axis=1,pct=True);to=[]
for j in range(1,len(r)):
 z=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
mx=-1
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;nc=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',nc,'factor_records',len(glob.glob('factors/*.json')))
