"""miner_1 one-idea validation: drawdown-weighted recovery acceleration, cutoff 2026-09-23."""
import os, json, glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2026-09-23')
H=(1,5,10,20)
prices={}; returns={}
for a in ASSETS:
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
    prices[a]=d['close'].astype(float); returns[a]=prices[a].pct_change()
# A recovery is most informative when it is an acceleration, but still below a 60-observation peak.
F={}; LIB={k:{} for k in ['miner_1_volnorm_reversal_5obs','miner_1_ravmom_20obs','miner_2_realized_volatility_20obs','miner_3_relative_volume_participation_20d','miner_3_risk_adjusted_trend_20d','miner_3_orthogonal_trend_acceleration_20_60obs','miner_3_negative_spx_beta_20obs']}
spxret=returns['SPX']
for a,p in prices.items():
    r=returns[a]; vol5=r.rolling(5,min_periods=4).std(); vol20=r.rolling(20,min_periods=15).std()
    dd=(p/p.rolling(60,min_periods=45).max()-1).clip(upper=0)
    accel=(p/p.shift(5)-1)-(p/p.shift(20)-1)/4
    F[a]=accel*(-dd)  # positive: a recent rebound accelerates while a material recovery opportunity remains
    LIB['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/vol5
    LIB['miner_1_ravmom_20obs'][a]=(p/p.shift(20)-1)/vol20
    LIB['miner_2_realized_volatility_20obs'][a]=-vol20
    # volume signal exactly follows admitted definition where volume is available; neutral missing values excluded
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
    if 'volume' in d and d.volume.notna().any(): LIB['miner_3_relative_volume_participation_20d'][a]=d.volume.astype(float)/d.volume.astype(float).rolling(20,min_periods=15).mean()
    else: LIB['miner_3_relative_volume_participation_20d'][a]=pd.Series(np.nan,index=p.index)
    LIB['miner_3_risk_adjusted_trend_20d'][a]=(p/p.shift(20)-1)/vol20
    LIB['miner_3_orthogonal_trend_acceleration_20_60obs'][a]=(p/p.shift(20)-1)-(p/p.shift(60)-1)/3
    beta=r.rolling(20,min_periods=15).cov(spxret)/spxret.rolling(20,min_periods=15).var()
    LIB['miner_3_negative_spx_beta_20obs'][a]=-beta
f=pd.DataFrame(F).sort_index()
fw={h:pd.DataFrame({a:p.shift(-h)/p-1 for a,p in prices.items()}).reindex(f.index) for h in H}
def ics(h):
    arr=[]; sizes=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt].rename('x'),fw[h].loc[dt].rename('y')],axis=1).dropna()
        if len(z)>=8: arr.append((dt,z.x.corr(z.y,method='spearman'))); sizes.append(len(z))
    return pd.Series(dict(arr)),np.mean(sizes)
print('FACTOR drawdown_weighted_recovery_acceleration_60_20_5 = (r5-r20/4)*max(rolling_max60/close-1,0)')
print('data_through',f.index.max().date(),'factor_date_range',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
RES={}
for h in H:
    x,n=ics(h); RES[h]=x
    print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} mean_n={n:.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
    for label,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
        q=x[mask]; print(f'  {label} n={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
ranks=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(ranks)):
    z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'coverage_cells={f.notna().sum().sum()}/{f.size} ({f.notna().stack().mean():.4f}) rank_turnover={np.mean(turn):.6f} turnover_dates={len(turn)}')
mx=0; ev={}
for name,parts in LIB.items():
    q=pd.concat([f.stack().rename('candidate'),pd.DataFrame(parts).stack().rename('library')],axis=1).dropna()
    rho=q.candidate.corr(q.library,method='spearman') if len(q)>2 else np.nan
    ev[name]=(rho,len(q)); mx=max(mx,abs(rho)) if np.isfinite(rho) else np.inf
    print(f'library {name} rho={rho:.6f} cells={len(q)}')
print(f'max_abs_library_correlation={mx:.6f}; library_json_records={len(glob.glob("factors/*.json"))}')
