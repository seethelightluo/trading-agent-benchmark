"""Miner_3 candidate: downside correlation stability, validated 2028-02-10.
Higher signal is low conditional co-movement with equal-weight market on negative days,
residualized against ordinary peer crowding and 20d realized volatility.
"""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-09')
def read(a,c='close',path='../persistent/stock_data/'):
 d=pd.read_csv(path+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=END,c],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A});R=P.pct_change(fill_method=None);m=R.mean(1);v=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std()
def res(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-15:
   b=z.x.cov(z.y)/z.y.var();o.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
raw=pd.DataFrame({a:R[a].where(m<0).rolling(30,min_periods=10).corr(m.where(m<0)) for a in A});F=res(res(-raw,peer),v)
mom=(P/P.shift(20)-1)/(v+1e-12)
def macro(n):return read(n,path='../persistent/index_data/').reindex(P.index).pct_change(fill_method=None)
def beta(n):return pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(macro(n)) for a in A})
def losscl():
 I=(R<0).astype(float);return pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
H=pd.DataFrame({a:read(a,'high') for a in A});Lo=pd.DataFrame({a:read(a,'low') for a in A});cl=(P-Lo)/(H-Lo).replace(0,np.nan);dd=P/P.rolling(20,min_periods=15).max()-1
L={'ravmom':mom,'reversal5':-(P/P.shift(5)-1)/(v5+1e-12),'realized_volatility':v,'peer_crowding':peer,'dxy_residual':res(beta('DXY'),peer),'vix_residual':res(beta('VIX'),peer),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'loss_clustering':losscl()}
vx=read('VIX',path='../persistent/index_data/').reindex(P.index);L['vix_crowding']=peer.mul((vx/vx.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
bd=pd.DataFrame({a:R[a].where(m<0).rolling(30,min_periods=10).cov(m.where(m<0))/m.where(m<0).rolling(30,min_periods=10).var() for a in A});bu=pd.DataFrame({a:R[a].where(m>0).rolling(30,min_periods=10).cov(m.where(m>0))/m.where(m>0).rolling(30,min_periods=10).var() for a in A});L['downside_beta_asymmetry']=-(bd-bu)
L['inverse_skew']=res(res(-R.rolling(20,min_periods=15).skew(),mom),v);L['close_location']=pd.DataFrame({a:cl[a].where(R[a]<0).rolling(20,min_periods=6).mean()-cl[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A});L['drawdown_inverse_ac']=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A});L['inverse_upside_resid']=-res(R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5),mom)
print('FACTOR downside_correlation_stability_30 visible_through',END.date(),'assets',len(A));ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; a=[];ns=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:a.append((t,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 s=pd.Series(dict(a),dtype=float);ics[h]=s;sd=s.std(ddof=1);print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} mean_instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(s)):.6f}')
for n,q in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_28',ics[10].index>='2026-01-01')]:
 s=ics[10][q];print('regime',n,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
r=F.rank(axis=1,pct=True);tr=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:tr.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(tr):.6f}')
mx=-1
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'active_factor_records',len([x for x in glob.glob('factors/*.json') if 'deprecated' not in x]))
