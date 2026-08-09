"""Miner_3: volatility-conditioned medium-horizon reversion residual validation."""
import numpy as np,pandas as pd,glob,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-09-22')
def load(path):
 d=pd.read_csv(path,parse_dates=['date']);d=d[d.date<=END].drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A});R=P.pct_change(fill_method=None)
def macro(n):return load('../persistent/index_data/'+n+'.csv').reindex(P.index)
def csres(x,y):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  z=pd.concat([x.loc[t].rename('x'),y.loc[t].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.y.var()>1e-16:
   b=z.x.cov(z.y)/z.y.var();out.loc[t,z.index]=z.x-(z.x.mean()-b*z.y.mean()+b*z.y)
 return out
v5=R.rolling(5,min_periods=4).std();v10=R.rolling(10,min_periods=8).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std()
mom20=(P/P.shift(20)-1)/v20
# Candidate: normalized ten-day price reversal, residualized each date versus 20d trend,
# and emphasized only when current short volatility exceeds an asset's normal volatility.
raw=-(P/P.shift(10)-1)/(v10+1e-12)
stress=(v5/(v60+1e-12)).clip(.5,2)
F=csres(raw,mom20)*stress
# Reconstruct all admitted library signal definitions for required correlation screen.
L={}
L['ravmom']=mom20;L['reversal5']=-(P/P.shift(5)-1)/(v5+1e-12);L['realized_vol']=v20
peer=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:peer[a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1)
L['peer_crowding']=peer
for fn,key in [('VIX','vix_resid'),('DXY','dxy_resid')]:
 m=macro(fn).pct_change(fill_method=None); x=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A});L[key]=csres(x,peer)
vix=macro('VIX');shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0)
L['vix_peer']=peer.mul(shock,axis=0)
L['low_vov']=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())
market=R.mean(axis=1); down=market<0;up=market>0
# beta difference computed only when each subset has >=6 observations in 30d window
bdown=pd.DataFrame(index=P.index,columns=A);bup=bdown.copy()
for a in A:
 bdown[a]=R[a].where(down).rolling(30,min_periods=6).cov(market.where(down))/market.where(down).rolling(30,min_periods=6).var()
 bup[a]=R[a].where(up).rolling(30,min_periods=6).cov(market.where(up))/market.where(up).rolling(30,min_periods=6).var()
L['downside_beta']=-(bdown-bup)
neg=R.where(R<0,0);pos=R.where(R>0,0)
up_part=pos.pow(2).rolling(20,min_periods=15).mean().pow(.5); L['inverse_upside_resid']=-csres(up_part,mom20)
# loss clustering
I=(R<0).astype(float);L['loss_clustering']=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift(1))/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
# inverse residual negative skew, then volatility residual
sk=-R.rolling(20,min_periods=15).skew();L['inverse_skew']=csres(csres(sk,mom20),v20)
# close location resilience
CL=pd.DataFrame(index=P.index,columns=A)
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']);d=d[d.date<=END].drop_duplicates('date').set_index('date').reindex(P.index)
 loc=(pd.to_numeric(d.close)-pd.to_numeric(d.low))/(pd.to_numeric(d.high)-pd.to_numeric(d.low)).replace(0,np.nan)
 CL[a]=loc.where(R[a]<0).rolling(20,min_periods=6).mean()-loc.where(R[a]>0).rolling(20,min_periods=6).mean()
L['close_location']=CL
# drawdown-conditioned inverse autocorrelation
DD=P/P.rolling(20,min_periods=15).max()-1; AC=pd.DataFrame(index=P.index,columns=A)
for a in A:
 q=R[a].where(DD[a]<0);AC[a]=-q.rolling(30,min_periods=12).corr(q.shift(1))
L['drawdown_inverse_ac']=AC
print('FACTOR volatility_stress_reversion_residual_10 = cs_residual[-return_10/vol_10 ~ ravmom_20] * clip(vol_5/vol_60,.5,2); visible_through',END.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;vals=[];n=[]
 for t in P.index:
  z=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.r,method='spearman')));n.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x;sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} mean_instruments={np.mean(n):.2f} se={sd/np.sqrt(len(x)):.6f}')
for lab,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',ics[10].index>='2026-01-01')]:
 x=ics[10][mask];print('regime',lab,'dates',len(x),'IC',None if len(x)==0 else round(x.mean(),6),'ICIR',None if len(x)==0 else round(x.mean()/x.std(ddof=1),6),'hit',None if len(x)==0 else round((x>0).mean(),4))
r=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(turns):.6f}')
mx=0
for k,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',k,'rho',f'{rho:.6f}','cells',len(z))
 if abs(rho)>mx:mx=abs(rho);who=k;nc=len(z)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',nc,'records',len(glob.glob('factors/*.json')))
PY
