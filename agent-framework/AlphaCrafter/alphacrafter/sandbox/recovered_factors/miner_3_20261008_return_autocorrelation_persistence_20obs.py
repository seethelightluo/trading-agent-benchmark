"""miner_3: One candidate--20-observation return-autocorrelation persistence.
Higher lag-1 serial correlation identifies assets whose recent daily moves have
persisted rather than alternated; test as a cross-asset leadership condition.
"""
import pandas as pd, numpy as np, glob, json, os
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=[1,5,10,20]; CUT=pd.Timestamp('2026-10-07')
def load(sym, root='../persistent/stock_data'):
 d=pd.read_csv(f'{root}/{sym}.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[d.index<=CUT,'close'].astype(float)
P={a:load(a) for a in ASSETS}; R={a:x.pct_change() for a,x in P.items()}
# factor: Pearson autocorrelation of daily returns at lag 1 inside trailing 20 obs
F=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in ASSETS})
FW={h:pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in ASSETS}) for h in H}
print('FACTOR return_autocorrelation_persistence_20obs = rolling_corr(r_t,r_t-1,20,min_periods=15)')
print('visible history',F.index.min().date(),F.index.max().date(),'assets',len(ASSETS))
def calc(h):
 vals=[]; coverage=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),FW[h].loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman'))); coverage.append(len(z)/len(ASSETS))
 x=pd.Series(dict(vals)); return x,float(np.mean(coverage))
allx={}
for h in H:
 x,c=calc(h);allx[h]=x
 print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} IC_se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={c:.4f}')
# report regimes at candidate's likely tactical horizon 5d
for h in H:
 print('regimes horizon',h)
 x=allx[h]
 for n,m in [('2020',x.index<'2021-01-01'),('2021_2022',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_2024',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_2026',x.index>='2025-01-01')]:
  y=x[m]; print(f' {n}: dates={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
ranks=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'mean_rank_turnover={np.mean(turns):.6f}; signal_cells={F.notna().sum().sum()}/{F.size} ({F.notna().mean().mean():.4f})')
# Reconstruct all admitted signal definitions; verify diversification on common signal cells.
vol=pd.DataFrame({a:R[a].rolling(20,min_periods=15).std() for a in ASSETS})
trend=pd.DataFrame({a:(P[a]/P[a].shift(20)-1)/vol[a] for a in ASSETS})
rev=pd.DataFrame({a:-(P[a]/P[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in ASSETS})
rv=pd.DataFrame({a:np.log(pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[lambda q:q.index<=CUT,'volume'].replace(0,np.nan)/pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[lambda q:q.index<=CUT,'volume'].replace(0,np.nan).rolling(20,min_periods=15).mean()) for a in ASSETS})
acc_raw=pd.DataFrame({a:(P[a]/P[a].shift(20)-P[a].shift(20)/P[a].shift(60))/vol[a] for a in ASSETS})
acc=pd.DataFrame(index=F.index,columns=ASSETS,dtype=float)
for dt in F.index:
 z=pd.concat([acc_raw.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  b=np.polyfit(z.t,z.a,1); acc.loc[dt,z.index]=z.a-(b[0]*z.t+b[1])
spxb=pd.DataFrame({a:-R[a].rolling(20,min_periods=15).cov(R['SPX'])/R['SPX'].rolling(20,min_periods=15).var().replace(0,np.nan) for a in ASSETS})
dxy=load('DXY','../persistent/index_data'); dr=dxy.pct_change(); dbv=dr.rolling(20,min_periods=15).var().replace(0,np.nan)
dxyb=pd.DataFrame({a:R[a].rolling(20,min_periods=15).cov(dr)/dbv for a in ASSETS})
libs={'risk_adjusted_trend':trend,'relative_volume':rv,'realized_volatility':vol,'ravmom':trend,'volnorm_reversal':rev,'orthogonal_acceleration':acc,'negative_spx_beta':spxb,'dxy_beta':dxyb}
mx=0; evidence={}
for n,l in libs.items():
 z=pd.concat([F.stack().rename('x'),l.stack().rename('y')],axis=1).dropna(); rho=z.x.corr(z.y,method='spearman'); evidence[n]=rho; mx=max(mx,abs(rho)); print(f'library_{n}_rho={rho:.6f}; cells={len(z)}')
print(f'library_files={len(glob.glob("factors/*.json"))}; max_abs_library_correlation={mx:.6f}')
