"""Miner_3 candidate: downside correlation stability, validated 2027-12-16.
Higher score means less co-movement with the cross-asset market on its negative days,
residualized cross-sectionally against ordinary peer crowding and 20d volatility.
"""
import numpy as np,pandas as pd,glob,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-15')
def read(a,c='close',path='../persistent/stock_data/'):
 d=pd.read_csv(path+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=END,c],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A}); R=P.pct_change(fill_method=None); m=R.mean(axis=1); v=R.rolling(20,min_periods=15).std()
def csres(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-15:
   b=z.x.cov(z.y)/z.y.var();o.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
# conditional rolling correlation uses only negative market days; min 10 pairs
raw=pd.DataFrame({a:R[a].where(m<0).rolling(30,min_periods=10).corr(m.where(m<0)) for a in A})
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
F=csres(csres(-raw,peer),v)
# Signals reconstructed from definition records; all admitted factors explicitly tested.
def macro(n):
 x=read(n,path='../persistent/index_data/');return x.reindex(P.index).pct_change(fill_method=None)
def beta(n): return pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(macro(n)) for a in A})
def losscl():
 I=(R<0).astype(float);return pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
L={}; mom=(P/P.shift(20)-1)/(v+1e-12); v5=R.rolling(5,min_periods=4).std(); L['ravmom']=mom;L['reversal5']=-(P/P.shift(5)-1)/(v5+1e-12);L['realized_volatility']=v;L['peer_crowding']=peer
L['dxy_residual']=csres(beta('DXY'),peer);L['vix_residual']=csres(beta('VIX'),peer); vx=read('VIX',path='../persistent/index_data/').reindex(P.index);L['vix_crowding']=peer.mul((vx/vx.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0);L['low_vov']=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())
bd=pd.DataFrame({a:R[a].where(m<0).rolling(30,min_periods=10).cov(m.where(m<0))/m.where(m<0).rolling(30,min_periods=10).var() for a in A});bu=pd.DataFrame({a:R[a].where(m>0).rolling(30,min_periods=10).cov(m.where(m>0))/m.where(m>0).rolling(30,min_periods=10).var() for a in A});L['downside_beta_asymmetry']=-(bd-bu)
L['loss_clustering']=losscl(); up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);L['inverse_upside_resid']=-csres(up,mom);L['inverse_skew']=csres(csres(-R.rolling(20,min_periods=15).skew(),mom),v)
H=pd.DataFrame({a:read(a,'high') for a in A});Lo=pd.DataFrame({a:read(a,'low') for a in A});loc=(P-Lo)/(H-Lo).replace(0,np.nan);L['close_location']=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A});dd=P/P.rolling(20,min_periods=15).max()-1;L['drawdown_inverse_ac']=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A})
# exact residual post-loss proxy
L['post_loss_rebound']=csres(csres((R.shift(-1).where(R<0)).rolling(20,min_periods=6).mean()/v,L['loss_clustering']),mom)
print('FACTOR downside_correlation_stability_30 visible_through',END.date(),'assets',len(A))
ICS={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; x=[]; nn=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:x.append((t,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 s=pd.Series(dict(x));ICS[h]=s;sd=s.std(ddof=1);print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} mean_instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(s)):.6f}')
for name,mask in [('2020_21',ICS[10].index<'2022-01-01'),('2022_23',(ICS[10].index>='2022-01-01')&(ICS[10].index<'2024-01-01')),('2024_25',(ICS[10].index>='2024-01-01')&(ICS[10].index<'2026-01-01')),('2026_27',ICS[10].index>='2026-01-01')]:
 s=ICS[10][mask];print('regime',name,'dates',len(s),'IC',round(s.mean(),6) if len(s) else None,'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else None,'hit',round((s>0).mean(),4) if len(s) else None)
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(turn):.6f}')
mx=-1
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'factor_records',len(glob.glob('factors/*.json')))
