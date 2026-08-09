"""miner_3: DXY-upside beta (one candidate).
The signal measures an asset's sensitivity specifically on strong-dollar days,
which may distinguish stress protection from unconditional DXY exposure.
"""
import pandas as pd, numpy as np, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2026-10-21'); H=[1,5,10,20]
def px(a,root='../persistent/stock_data'):
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=CUT,'close'],errors='coerce')
P={a:px(a) for a in A}; R={a:P[a].pct_change(fill_method=None) for a in A}
dxy=px('DXY','../persistent/index_data'); dr=dxy.pct_change(fill_method=None)
# Candidate: negative 60-observation OLS beta, retaining only the largest 30% positive DXY daily moves.
# A high value is lower sensitivity to abrupt dollar-strengthening shocks.
up=dr.where(dr>=dr.rolling(60,min_periods=40).quantile(.70))
F=pd.DataFrame({a:-R[a].rolling(60,min_periods=40).cov(up)/up.rolling(60,min_periods=40).var().replace(0,np.nan) for a in A})
FW={h:pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in A}) for h in H}
print('FACTOR dxy_upside_defensiveness_60obs = -cov(r_asset, r_DXY | r_DXY >= rolling_60_q70)/var(r_DXY | same); rolling 60, min 40')
print('visible_through',CUT.date(),'history',F.index.min().date(),F.index.max().date(),'assets',len(A))
def ic(fw):
 out=[]; cov=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 return pd.Series(dict(out)),np.mean(cov)
ics={}
for h in H:
 x,c=ic(FW[h]);ics[h]=x
 print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} IC_se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for n,m in [('2020',x.index<'2021-01-01'),('2021_2022',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_2024',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_2026',x.index>='2025-01-01')]:
  y=x[m];print(f' regime={n} dates={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
ranks=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'mean_rank_turnover={np.mean(to):.6f}; signal_cells={F.notna().sum().sum()}/{F.size} ({F.notna().mean().mean():.4f})')
# Exact/reconstructed admitted-library signals for required pooled-cell diversification evidence.
vol=pd.DataFrame({a:R[a].rolling(20,min_periods=15).std() for a in A})
trend=pd.DataFrame({a:(P[a]/P[a].shift(20)-1)/vol[a] for a in A})
rev=pd.DataFrame({a:-(P[a]/P[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in A})
rv={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
 v=pd.to_numeric(d['volume'],errors='coerce').replace(0,np.nan);rv[a]=np.log(v/v.rolling(20,min_periods=15).mean())
rv=pd.DataFrame(rv)
raw=pd.DataFrame({a:(P[a]/P[a].shift(20)-P[a].shift(20)/P[a].shift(60))/vol[a] for a in A});acc=pd.DataFrame(index=F.index,columns=A,dtype=float)
for dt in F.index:
 z=pd.concat([raw.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  b=np.polyfit(z.t,z.a,1);acc.loc[dt,z.index]=z.a-(b[0]*z.t+b[1])
spxb=pd.DataFrame({a:-R[a].rolling(20,min_periods=15).cov(R['SPX'])/R['SPX'].rolling(20,min_periods=15).var().replace(0,np.nan) for a in A})
dv=dr.rolling(20,min_periods=15).var().replace(0,np.nan);dxyb=pd.DataFrame({a:R[a].rolling(20,min_periods=15).cov(dr)/dv for a in A})
libs={'risk_adjusted_trend':trend,'relative_volume':rv,'realized_volatility':vol,'ravmom':trend,'volnorm_reversal':rev,'orthogonal_acceleration':acc,'negative_spx_beta':spxb,'dxy_beta':dxyb}
mx=0; who=''
for n,l in libs.items():
 z=pd.concat([F.stack().rename('candidate'),l.stack().rename('library')],axis=1).dropna();rho=z.candidate.corr(z.library,method='spearman')
 print(f'library_{n}_rho={rho:.6f}; cells={len(z)}')
 if abs(rho)>mx:mx=abs(rho);who=n
print(f'library_files={len(glob.glob("factors/*.json"))}; max_abs_library_correlation={mx:.6f}; max_name={who}')
