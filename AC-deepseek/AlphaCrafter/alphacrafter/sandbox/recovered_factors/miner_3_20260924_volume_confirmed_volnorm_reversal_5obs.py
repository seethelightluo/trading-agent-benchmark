"""miner_3: one idea -- volume-confirmed, volatility-normalized 5-day reversal."""
import pandas as pd, numpy as np, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; END=pd.Timestamp('2026-09-23')
P={};R={};V={};FW={h:{} for h in H}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); R[a]=P[a].pct_change(); V[a]=d.volume.astype(float).replace(0,np.nan)
 for h in H: FW[h][a]=P[a].shift(-h)/P[a]-1
F={};TREND={};REV={};RV={};VOL={};ACC={};SPXB={}
for a in A:
 VOL[a]=R[a].rolling(20,min_periods=15).std()
 r5=P[a]/P[a].shift(5)-1; v5=R[a].rolling(5,min_periods=4).std()
 RV[a]=np.log(V[a]/V[a].rolling(20,min_periods=15).mean())
 # A reversal is trusted only when volume is not abnormally thin; clip confirmation to prevent volume outliers.
 confirm=RV[a].clip(lower=-1.0,upper=1.0)
 F[a]=-(r5/v5)*confirm
 TREND[a]=(P[a]/P[a].shift(20)-1)/VOL[a]
 REV[a]=-r5/v5
 ACC[a]=(P[a]/P[a].shift(20)-P[a].shift(20)/P[a].shift(60))/VOL[a]
 SPXB[a]=-R[a].rolling(20,min_periods=15).cov(R['SPX'])/R['SPX'].rolling(20,min_periods=15).var().replace(0,np.nan).reindex(R[a].index)
f=pd.DataFrame(F); trend=pd.DataFrame(TREND); acc=pd.DataFrame(ACC)
orth=pd.DataFrame(index=f.index,columns=A,dtype=float)
for dt in f.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  q=np.polyfit(z.t,z.a,1); orth.loc[dt,z.index]=z.a-(q[0]*z.t+q[1])
lib={'risk_adjusted_trend':trend,'relative_volume':pd.DataFrame(RV),'realized_volatility':pd.DataFrame(VOL),'ravmom':trend,'volnorm_reversal':pd.DataFrame(REV),'orthogonal_acceleration':orth,'negative_spx_beta':pd.DataFrame(SPXB)}
print('FACTOR volume_confirmed_volnorm_reversal_5obs = -(close/lag5(close)-1)/std_5(r) * clip(log(volume/mean_20(volume)),-1,1)')
print('visible_through',END.date(),'history',f.index.min().date(),f.index.max().date(),'assets',len(A))
def metrics(h):
 fw=pd.DataFrame(FW[h]);out=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.x.corr(z.y,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(out));return x,np.mean(cov)
for h in H:
 x,c=metrics(h);print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} IC_se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 if h==5:
  for n,m in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
   y=x[m];print(f'  {n}: n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'mean_rank_turnover={np.mean(turn):.6f}; signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0;who=''
for n,l in lib.items():
 z=pd.concat([f.stack().rename('x'),l.stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n
 print(f'library_{n}_rho={rho:.6f}; cells={len(z)}')
print(f'library_files={len(glob.glob("factors/*.json"))}; max_abs_library_correlation={mx:.6f}; highest={who}')
