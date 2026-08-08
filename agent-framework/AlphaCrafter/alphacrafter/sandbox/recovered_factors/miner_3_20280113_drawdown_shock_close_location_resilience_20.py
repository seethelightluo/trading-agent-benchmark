"""Miner_3: drawdown-shock close-location resilience. Visible data only through 2028-01-12."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-01-12')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=END,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(1); v=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std(); dd=P/P.rolling(20,min_periods=15).max()-1
H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A}); cl=(P-Lo)/(H-Lo).replace(0,np.nan)
def csres(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-15:
   b=z.x.cov(z.y)/z.y.var();o.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
mom=(P/P.shift(20)-1)/(v+1e-12)
# On loss days after a material 20d drawdown and a vol acceleration, closing near high indicates resilient absorption.
cond=(R<0)&(dd<-0.03)&(v5/v.replace(0,np.nan)>1.10)
raw=pd.DataFrame({a:cl[a].where(cond[a]).rolling(20,min_periods=4).mean() for a in A})
F=csres(csres(raw,mom),v)
# Reconstruct admitted factors with observable definitions for full-library orthogonality.
def macro(n):return rd(n,root='../persistent/index_data/').reindex(P.index).pct_change(fill_method=None)
def beta(n):return pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(macro(n)) for a in A})
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
def losscl():
 I=(R<0).astype(float);return pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
L={'ravmom':mom,'reversal5':-(P/P.shift(5)-1)/(v5+1e-12),'realized_volatility':v,'peer_crowding':peer,'dxy_residual':csres(beta('DXY'),peer),'vix_residual':csres(beta('VIX'),peer),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())}
vx=rd('VIX',root='../persistent/index_data/').reindex(P.index);L['vix_crowding']=peer.mul((vx/vx.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
bd=pd.DataFrame({a:R[a].where(M<0).rolling(30,min_periods=10).cov(M.where(M<0))/M.where(M<0).rolling(30,min_periods=10).var() for a in A});bu=pd.DataFrame({a:R[a].where(M>0).rolling(30,min_periods=10).cov(M.where(M>0))/M.where(M>0).rolling(30,min_periods=10).var() for a in A});L['downside_beta_asymmetry']=-(bd-bu);L['loss_clustering']=losscl();L['inverse_skew']=csres(csres(-R.rolling(20,min_periods=15).skew(),mom),v)
loc=pd.DataFrame({a:cl[a].where(R[a]<0).rolling(20,min_periods=6).mean()-cl[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A});L['close_location']=loc;L['drawdown_inverse_ac']=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A});L['inverse_upside_resid']=-csres(R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5),mom)
print('FACTOR drawdown_shock_close_location_resilience_20 visible_through',END.date(),'assets',len(A)); ICS={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];nn=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: out.append((t,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 s=pd.Series(dict(out));ICS[h]=s;sd=s.std(ddof=1);print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} mean_instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(s)):.6f}')
for n,q in [('2020_21',ICS[10].index<'2022-01-01'),('2022_23',(ICS[10].index>='2022-01-01')&(ICS[10].index<'2024-01-01')),('2024_25',(ICS[10].index>='2024-01-01')&(ICS[10].index<'2026-01-01')),('2026_28',ICS[10].index>='2026-01-01')]:
 s=ICS[10][q];print('regime',n,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else None,'hit',round((s>0).mean(),4))
r=F.rank(axis=1,pct=True);turn=[1-pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna().corr(method='spearman').iloc[0,1] for i in range(1,len(r)) if len(pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna())>=8];print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(turn):.6f}')
mx=-1
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'factor_records',len(glob.glob('factors/*.json')))
