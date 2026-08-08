import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];E=pd.Timestamp('2027-05-05')
def sv(p,c='close'):
 d=pd.read_csv(p,parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[c],errors='coerce')
P=pd.DataFrame({a:sv('../persistent/stock_data/'+a+'.csv') for a in A});V=pd.DataFrame({a:sv('../persistent/stock_data/'+a+'.csv','volume') for a in A});R=P.pct_change(fill_method=None);rv=V/V.rolling(60,min_periods=40).mean();F=rv.where(R>0).rolling(20,min_periods=4).mean()-rv.where(R<0).rolling(20,min_periods=4).mean();peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def resid(x,y):
 o=pd.DataFrame(index=x.index,columns=A)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.y.var()>0:
   b=q.x.cov(q.y)/q.y.var();o.loc[d,q.index]=q.x-(q.x.mean()-b*q.y.mean()+b*q.y)
 return o
def beta(fn):
 m=sv('../persistent/index_data/'+fn).reindex(P.index).pct_change(fill_method=None);return resid(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),peer)
v=sv('../persistent/index_data/VIX.csv').reindex(P.index);sh=(v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0);up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
L={'vix_beta_residual_peer20':beta('VIX.csv'),'dxy_beta_residual_peer20':beta('DXY.csv'),'vix_conditioned_peer_crowding':peer.mul(sh,axis=0),'high_vix_momentum_residual_downside_asymmetry':resid(np.log((up+1e-8)/(dn+1e-8)),P/P.shift(20)-1).mul(sh,axis=0)}
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();print(n,'rho=%.6f'%q.f.corr(q.x,method='spearman'),'common_cells',len(q))
