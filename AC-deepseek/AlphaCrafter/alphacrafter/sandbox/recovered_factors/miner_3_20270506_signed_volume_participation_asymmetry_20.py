"""miner_3: signed volume participation asymmetry; one interpretable liquidity/conviction factor."""
import numpy as np,pandas as pd,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-05-05')
def dat(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.close,errors='coerce'),pd.to_numeric(d.volume,errors='coerce')
def ix(fn):
 d=pd.read_csv('../persistent/index_data/'+fn,parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.close,errors='coerce')
z={a:dat(a) for a in A};P=pd.DataFrame({a:z[a][0] for a in A});V=pd.DataFrame({a:z[a][1] for a in A});R=P.pct_change(fill_method=None);M=R.mean(axis=1);v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std()
# Conviction: relative volume accompanying positive returns less that accompanying negative returns.
# Each leg is normalized by its own trailing 60d volume baseline, preventing size effects.
rv=V/V.rolling(60,min_periods=40).mean(); F=rv.where(R>0).rolling(20,min_periods=8).mean()-rv.where(R<0).rolling(20,min_periods=8).mean()
def resid(x,y):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.y.var()>0:
   b=q.x.cov(q.y)/q.y.var();o.loc[d,q.index]=q.x-(q.x.mean()-b*q.y.mean()+b*q.y)
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def mb(fn):
 m=ix(fn).reindex(P.index).pct_change(fill_method=None);return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),peer)
vv=ix('VIX.csv').reindex(P.index); shock=(vv/vv.rolling(60,min_periods=40).mean()-1).clip(lower=0);up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
def cb(a,mask): return R[a].where(mask).rolling(30,min_periods=10).cov(M.where(mask))/M.where(mask).rolling(30,min_periods=10).var()
ind=(R<0).astype(float);loss=ind.rolling(20,min_periods=12).cov(ind.shift(1))/(ind.rolling(20,min_periods=12).mean()*(1-ind.rolling(20,min_periods=12).mean())+1e-12)
L={'ravmom':(P/P.shift(20)-1)/v20,'reversal':-(P/P.shift(5)-1)/v5,'realized_vol':v20,'peer':peer,'vix_resid':mb('VIX.csv'),'dxy_resid':mb('DXY.csv'),'high_vix_asym':resid(np.log((up+1e-8)/(dn+1e-8)),P/P.shift(20)-1).mul(shock,axis=0),'vix_peer':peer.mul(shock,axis=0),'low_vov':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'compression':-v5/(v60+1e-12),'downside_beta_asym':pd.DataFrame({a:-(cb(a,M<0)-cb(a,M>0)) for a in A}),'loss_clustering':loss}
print('FACTOR signed_volume_participation_asymmetry_20; visible',END.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 out=[];nn=[];fw=P.shift(-h)/P-1
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((d,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(nn):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
x=ics[10]
for n,m in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026_27',x.index>='2026-01-01')]:
 q=x[m];print(n,'dates',len(q),'IC',f'{q.mean():.6f}','ICIR',f'{q.mean()/q.std(ddof=1):.6f}','hit',f'{(q>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('signal_cell_coverage',f'{F.notna().mean().mean():.6f}','mean_daily_rank_turnover',f'{np.mean(to):.6f}')
mx=-1
for n,s in L.items():
 q=pd.concat([F.stack().rename('f'),s.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library',n,'rho',f'{rho:.6f}','common_cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells,'active_library_records',len(glob.glob('factors/*.json')))
