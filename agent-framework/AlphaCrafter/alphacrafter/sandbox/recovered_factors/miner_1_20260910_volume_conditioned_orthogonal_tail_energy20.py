"""miner_1: validate volume-conditioned trend-orthogonal downside energy, 20 observations."""
import glob, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base={}; tr={}; rel={}; price={}; libs={k:{} for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs','cross_asset_beta_compression_20obs','risk_adjusted_trend_acceleration_20_60d']}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); p=d.close.astype(float); v=d.volume.astype(float); r=p.pct_change()
 de=r.clip(upper=0).pow(2).rolling(20,min_periods=15).sum()/r.pow(2).rolling(20,min_periods=15).sum()
 t=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(); q=np.log(v/v.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
 # one interpretable idea: downside energy is informative only to extent it occurs amid above-normal participation
 base[a]=de*q.clip(lower=0); tr[a]=t; rel[a]=q; price[a]=p
 libs['miner_3_risk_adjusted_trend_20d'][a]=t; libs['miner_3_relative_volume_participation_20d'][a]=q
 libs['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); libs['miner_2_realized_volatility_20obs'][a]=r.rolling(20,min_periods=15).std()
 # beta to equal-weight cross-asset return, then negative beta (compression/defensiveness)
 # filled after panel construction
base=pd.DataFrame(base); T=pd.DataFrame(tr); Q=pd.DataFrame(rel); P=pd.DataFrame(price)
R=P.pct_change(); market=R.mean(axis=1); beta=R.rolling(20,min_periods=15).cov(market).div(market.rolling(20,min_periods=15).var(),axis=0)
for a in A:
 libs['cross_asset_beta_compression_20obs'][a]=-beta[a]
 libs['risk_adjusted_trend_acceleration_20_60d'][a]=((P[a]/P[a].shift(20)-1)-(P[a]/P[a].shift(60)-1))/R[a].rolling(20,min_periods=15).std()
# residualise candidate on both trend and raw participation cross-sectionally, intercept included
F=pd.DataFrame(np.nan,index=P.index,columns=A)
for dt in F.index:
 z=pd.concat([base.loc[dt].rename('y'),T.loc[dt].rename('trend'),Q.loc[dt].rename('vol')],axis=1).dropna()
 if len(z)>=8:
  b=np.linalg.lstsq(np.c_[np.ones(len(z)),z[['trend','vol']]],z.y,rcond=None)[0]; F.loc[dt,z.index]=z.y-np.c_[np.ones(len(z)),z[['trend','vol']]]@b
print('FACTOR volume_conditioned_trend_orthogonal_downside_energy_20obs: downside squared-return energy share x positive log relative-volume, residualized daily cross-sectionally on 20d risk-adjusted trend and relative volume')
print('history',P.index.min().date(),P.index.max().date(),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',round(F.notna().mean().mean(),6))
def calc(h):
 fw=P.shift(-h)/P-1; vals=[]; cov=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.r,method='spearman')));cov.append(len(z)/15)
 x=pd.Series(dict(vals)); sd=x.std(ddof=1)
 print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} coverage={np.mean(cov):.4f}')
 for label,mask in [('2020',x.index<'2021'),('2021-22',(x.index>='2021')&(x.index<'2023')),('2023-24',(x.index>='2023')&(x.index<'2025')),('2025-26',x.index>='2025')]:
  y=x[mask]; print(f' {label} n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f}')
for h in [1,5,10,20]:calc(h)
ranks=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(F)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',round(float(np.mean(turns)),6))
mx=0
for name,dct in libs.items():
 old=pd.DataFrame(dct).reindex(F.index); z=pd.concat([F.stack().rename('new'),old.stack().rename('old')],axis=1).dropna(); rho=z.new.corr(z.old,method='spearman'); mx=max(mx,abs(rho));print('LIB',name,'rho',f'{rho:.6f}','cells',len(z))
print('max_abs_library_correlation',f'{mx:.6f}','library_json_count',len(glob.glob('factors/*.json')))
