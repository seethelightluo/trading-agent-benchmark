"""miner_3 one-idea validation: 20-day return serial-dependence factor."""
import glob, json
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
H=(1,5,10,20)
F={}; FW={}; LIB={}
for a in ASSETS:
    d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
    p=d.close.astype(float); r=p.pct_change()
    # Lag-1 autocorrelation of returns: high values denote persistent (rather than alternating) price changes.
    F[a]=r.rolling(20,min_periods=15).corr(r.shift(1))
    FW[a]={h:p.shift(-h)/p-1 for h in H}
    LIB.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
    LIB.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(d.volume.astype(float)/d.volume.astype(float).rolling(20,min_periods=15).mean())
    LIB.setdefault('miner_1_ravmom_20obs',{})[a]=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
    LIB.setdefault('miner_2_realized_volatility_20obs',{})[a]=r.rolling(20,min_periods=15).std()
    LIB.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
f=pd.DataFrame(F).sort_index()
print('FACTOR: return_serial_dependence_20d = rolling_corr(return_t, return_t-1, 20); high identifies assets whose daily moves have recently persisted rather than reversed')
print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def stats(x):
    return (x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),x.std(ddof=1)/np.sqrt(len(x)))
def calc(h):
    fw=pd.DataFrame({a:FW[a][h] for a in ASSETS}); out=[]; cov=[]
    for dt in f.index:
        z=pd.DataFrame({'s':f.loc[dt],'r':fw.loc[dt]}).dropna()
        if len(z)>=8: out.append((dt,z.s.corr(z.r,method='spearman'))); cov.append(len(z)/15)
    return pd.Series(dict(out)),np.mean(cov)
for h in H:
    x,c=calc(h); m,ir,hit,se=stats(x)
    print(f'h={h} dates={len(x)} meanIC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} IC_se={se:.6f} coverage={c:.4f}')
    for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
        z=x[mask]; a,b,c1,d=stats(z); print(f'  {nm}: n={len(z)} IC={a:.6f} ICIR={b:.6f} hit={c1:.4f}')
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
    z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
    if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(turns):.6f}; valid_signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0
for name,vals in LIB.items():
    z=pd.concat([f.stack().rename('a'),pd.DataFrame(vals).stack().rename('b')],axis=1).dropna(); rho=z.a.corr(z.b,method='spearman'); mx=max(mx,abs(rho)); print(f'library {name}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len(glob.glob("factors/*.json"))} max_abs_library_correlation={mx:.6f}')
