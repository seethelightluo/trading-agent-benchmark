"""miner_1 one idea: 20-observation return serial-dependence persistence.
Higher score is the lag-one autocorrelation of daily returns within the trailing
20 observations. It measures whether an asset's moves have recently persisted
rather than alternated, independently of its return level or volatility.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']; prices={}; volumes={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    d=d.set_index('date').sort_index()
    prices[a]=pd.to_numeric(d['close'],errors='coerce'); volumes[a]=pd.to_numeric(d['volume'],errors='coerce')
C=pd.DataFrame(prices).sort_index(); V=pd.DataFrame(volumes).reindex(C.index); R=C.pct_change(fill_method=None)
# rolling autocorr is intentionally a single, interpretable serial-dependence idea
F=R.rolling(20,min_periods=15).corr(R.shift(1))
LIB={
 'miner_3_risk_adjusted_trend_20d':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
 'miner_1_ravmom_20obs':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
 'miner_1_volnorm_reversal_5obs':-C.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(),
 'miner_2_volscaled_reversal_1obs':-R/R.rolling(20,min_periods=15).std(),
 'miner_3_relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan),
 'miner_3_quiet_trend_path_efficiency_20_60':(C.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum())*(1-R.rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)),
}
def rho(x,y):
 z=pd.concat([x,y],axis=1).dropna()
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
def met(h, mask=None):
 fw=C.shift(-h)/C-1; ix=F.index if mask is None else F.index[mask]
 ic=pd.Series([rho(F.loc[t],fw.loc[t]) for t in ix],index=ix).dropna()
 rank=F.rank(axis=1,pct=True); turn=rank.diff().abs().stack().mean()
 return {'dates':len(ic),'ic':ic.mean(),'icir':ic.mean()/ic.std(ddof=1),'hit':(ic>0).mean(), 'mean_names':F.loc[ix].notna().sum(axis=1).mean(), 'turnover':turn}
print('FACTOR return_serial_dependence_persistence_20obs')
print('endpoint',C.index.max().date(),'period',C.index.min().date(),C.index.max().date(),'assets',len(assets))
print('cells',int(F.notna().sum().sum()),'of',F.size,'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 m=met(h);print(f"H={h} dates={m['dates']} IC={m['ic']:.6f} ICIR={m['icir']:.6f} hit={m['hit']:.6f} names={m['mean_names']:.2f}")
for label,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021_2022','2021-01-01','2022-12-31'),('2023_2024','2023-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31')]:
 mask=(F.index>=lo)&(F.index<=hi);m=met(1,mask); print(f"REGIME {label} dates={m['dates']} IC={m['ic']:.6f} ICIR={m['icir']:.6f} hit={m['hit']:.6f}")
print('TURNOVER',met(1)['turnover'])
mx=-1; who=''
for n,X in LIB.items():
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna(); val=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 print('LIB',n,'rho',f'{val:.6f}','cells',len(z))
 if abs(val)>mx: mx=abs(val);who=n
print('MAX_ABS_LIBRARY_CORRELATION',f'{mx:.6f}',who)
