"""miner_1 one idea: 20-observation upside/downside return-capture ratio.
Score is cumulative positive daily return magnitude divided by cumulative
negative-return magnitude across the trailing 20 observations.  Higher values
identify assets whose recent path had favorable asymmetric participation rather
than merely a high total return.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; P={}; Q={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
 P[a]=pd.to_numeric(d['close'],errors='coerce'); Q[a]=pd.to_numeric(d['volume'],errors='coerce')
C=pd.DataFrame(P).sort_index(); V=pd.DataFrame(Q).reindex(C.index); R=C.pct_change(fill_method=None)
pos=R.clip(lower=0).rolling(20,min_periods=15).sum(); neg=(-R.clip(upper=0)).rolling(20,min_periods=15).sum()
F=(pos/(neg+1e-8)).replace([np.inf,-np.inf],np.nan)
LIB={
'risk_adjusted_trend_20d':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
'ravmom_20obs':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
'volnorm_reversal_5obs':-C.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(),
'volscaled_reversal_1obs':-R/R.rolling(20,min_periods=15).std(),
'relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan),
'quiet_trend_path_efficiency_20_60':(C.pct_change(20,fill_method=None).abs()/R.abs().rolling(20,min_periods=15).sum())*(1-R.rolling(20,min_periods=15).std().rolling(60,min_periods=40).rank(pct=True)),}
def csrho(x,y):
 z=pd.concat([x,y],axis=1).dropna()
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
def metric(h,sel=None):
 fw=C.shift(-h).div(C).sub(1); dates=F.index if sel is None else F.index[sel]
 ics=pd.Series([csrho(F.loc[t],fw.loc[t]) for t in dates],index=dates).dropna()
 ranks=F.rank(axis=1,pct=True); turnover=ranks.diff().abs().stack().mean()
 return len(ics),ics.mean(),ics.mean()/ics.std(ddof=1) if len(ics)>1 else np.nan,(ics>0).mean(),F.loc[dates].notna().sum(axis=1).mean(),turnover
print('FACTOR upside_downside_capture_ratio_20obs'); print('endpoint',C.index.max().date(),'period',C.index.min().date(),C.index.max().date(),'assets',len(assets))
print('cells',int(F.notna().sum().sum()),'of',F.size,'coverage',round(F.notna().mean().mean(),6))
for h in (1,5,10,20):
 a=metric(h); print(f'H={h} dates={a[0]} IC={a[1]:.6f} ICIR={a[2]:.6f} hit={a[3]:.6f} names={a[4]:.2f}')
for label,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021_2022','2021-01-01','2022-12-31'),('2023_2024','2023-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31')]:
 a=metric(1,(F.index>=lo)&(F.index<=hi));print(f'REGIME {label} dates={a[0]} IC={a[1]:.6f} ICIR={a[2]:.6f} hit={a[3]:.6f}')
print('TURNOVER',metric(1)[5])
mx=-1;who=''
for n,X in LIB.items():
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna(); r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 print('LIB',n,'rho',f'{r:.6f}','cells',len(z))
 if abs(r)>mx: mx=abs(r);who=n
print('MAX_ABS_LIBRARY_CORRELATION',f'{mx:.6f}',who)
