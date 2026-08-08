"""One idea: cross-asset dispersion-conditioned idiosyncratic stability.
On days when cross-sectional absolute-return dispersion is above its trailing
median, rank assets by negative 10-day idiosyncratic return volatility (asset
volatility after subtracting the equal-weight cross-asset daily return). This
is a conditional defensive-quality factor, distinct from raw low volatility and
price reversal because it targets asset-specific instability during fragmented
markets."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce')
C=pd.DataFrame(P).sort_index(); V=pd.DataFrame(V).reindex(C.index); R=C.pct_change(fill_method=None)
# Market fragmentation is cross-sectional sd of contemporaneous absolute moves.
disp=R.abs().std(axis=1,ddof=1)
active=disp>disp.rolling(60,min_periods=40).median()
# Remove same-day equal-weight move, then favor lower trailing idiosyncratic variance.
idem=R.sub(R.mean(axis=1),axis=0)
F=-idem.rolling(10,min_periods=8).std().where(active, np.nan)
# Signals of every admitted/non-deprecated library factor; aliases retained for exact overlap.
LIB={
'miner_3_risk_adjusted_trend_20d':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
'miner_1_ravmom_20obs':C.pct_change(20,fill_method=None)/R.rolling(20,min_periods=15).std(),
'miner_1_volnorm_reversal_5obs':-C.pct_change(5,fill_method=None)/R.rolling(5,min_periods=4).std(),
'miner_2_volscaled_reversal_1obs':-R/R.rolling(20,min_periods=15).std(),
'miner_3_relative_volume_participation_20d':np.log(V/V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan),
}
def daily_spear(x,y):
 z=pd.concat([x,y],axis=1).dropna()
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
def metrics(h, subset=None):
 fw=C.shift(-h)/C-1; dates=F.index if subset is None else F.loc[subset].index; ics=[]; ns=[]
 for t in dates:
  q=daily_spear(F.loc[t],fw.loc[t])
  if pd.notna(q): ics.append(q); ns.append(F.loc[t].notna().sum())
 x=np.array(ics); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)
print('FACTOR dispersion_conditioned_idiosyncratic_stability_10obs')
print('endpoint',C.index.max().date(),'range',C.index.min().date(),C.index.max().date(),'assets',len(A))
print('active_dates',int(active.sum()),'of',len(active),'signal_cells',int(F.notna().sum().sum()),'of',F.size,'coverage',round(F.notna().mean().mean(),6))
for h in (1,5,10,20):
 n,ic,ir,hit,mn=metrics(h);print(f'H={h} dates={n} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.6f} mean_names={mn:.2f}')
for name,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31')]:
 n,ic,ir,hit,mn=metrics(1,(F.index>=lo)&(F.index<=hi));print(f'REGIME {name} h1_dates={n} IC={ic:.6f} ICIR={ir:.6f} hit={hit:.6f} mean_names={mn:.2f}')
ranks=F.rank(axis=1,pct=True); turn=ranks.diff().abs().stack().mean();print('TURNOVER mean_abs_rank_change',round(turn,6))
mx=-1; who=''
for name,X in LIB.items():
 q=pd.concat([F.stack(),X.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 # daily max shown for diagnostic; admission value is pooled signal-cell Spearman.
 ds=[daily_spear(F.loc[t],X.loc[t]) for t in F.index]; ds=[x for x in ds if pd.notna(x)]
 print('LIBCORR',name,'cells',len(q),'pooled',round(rho,6),'daily_max_abs',round(max(map(abs,ds)),6) if ds else None)
 if abs(rho)>mx:mx=abs(rho);who=name
print('MAX_LIBRARY_CORRELATION',round(mx,6),who)
"""
