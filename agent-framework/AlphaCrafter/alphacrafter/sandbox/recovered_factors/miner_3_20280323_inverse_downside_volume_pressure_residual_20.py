"""Miner_3 single-factor study: inverse downside-volume pressure residual, data through 2028-03-22."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];E=pd.Timestamp('2028-03-22')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); V=pd.DataFrame({a:rd(a,'volume') for a in A});R=P.pct_change(fill_method=None); v=R.rolling(20,min_periods=15).std()
# Novel liquidity/participation construct: log-volume anomaly on loss days minus gain days. Negative sign favors assets not experiencing disproportionate selling participation.
lv=np.log(V.replace(0,np.nan)); an=lv-lv.rolling(20,min_periods=15).mean(); raw=pd.DataFrame({a:-(an[a].where(R[a]<0).rolling(20,min_periods=6).mean()-an[a].where(R[a]>0).rolling(20,min_periods=6).mean()) for a in A})
def res(x,*cs):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna();X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
F=res(raw,v,peer)
# Conservative orthogonality diagnostic against major risk/participation factors; full library comparison is performed before any persistence.
mom=(P/P.shift(20)-1)/(v+1e-12);v5=R.rolling(5,min_periods=4).std(); loss=pd.DataFrame({a:(R[a]<0).astype(float).rolling(20,min_periods=15).cov((R[a]<0).astype(float).shift()) for a in A})
L={'risk_adjusted_momentum':mom,'realized_volatility':v,'peer_crowding':peer,'loss_clustering_proxy':loss,'low_volatility_of_volatility':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())}
print('FACTOR inverse_downside_volume_pressure_residual_20 visible_through',E.date(),'assets',len(A))
ics={}
for h in [1,5,10,20]:
 out=[];nn=[];fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,m in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',(ics[10].index>='2026-01-01')&(ics[10].index<'2028-01-01')),('2028_ytd',ics[10].index>='2028-01-01')]:
 x=ics[10][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else 'nan','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();print('diagnostic_rho',n,f'{q.f.corr(q.x,method="spearman"):.6f}','cells',len(q))
