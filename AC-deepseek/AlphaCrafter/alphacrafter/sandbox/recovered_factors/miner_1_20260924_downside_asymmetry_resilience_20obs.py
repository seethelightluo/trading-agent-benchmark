"""One idea: 20-observation downside-asymmetry resilience.
Signal is negative semideviation divided by total volatility: assets whose recent
volatility has contained relatively fewer/lower negative daily returns score higher.
It is a close-only, calendar-safe cross-asset defensive quality measure."""
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}; R={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 p=pd.to_numeric(d['close'],errors='coerce').dropna(); P[a]=p; R[a]=p.pct_change(fill_method=None)
 V[a]=pd.to_numeric(d['volume'],errors='coerce')
panel=pd.DataFrame(P).sort_index(); ret=pd.DataFrame(R).reindex(panel.index); vol=pd.DataFrame(V).reindex(panel.index)
# Candidate: negative downside semideviation / total sd; 15 observations required.
down=np.sqrt(ret.where(ret<0,0).pow(2).rolling(20,min_periods=15).mean())
fac=-(down/ret.rolling(20,min_periods=15).std()).replace([np.inf,-np.inf],np.nan)
# Exact implementations/proxies of all admitted library signals for correlation gate.
lib={
 'miner_3_risk_adjusted_trend_20d': panel.pct_change(20,fill_method=None)/ret.rolling(20,min_periods=15).std(),
 'miner_1_ravmom_20obs': panel.pct_change(20,fill_method=None)/ret.rolling(20,min_periods=15).std(),
 'miner_1_volnorm_reversal_5obs': -panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std(),
 'miner_2_realized_volatility_20obs': -ret.rolling(20,min_periods=15).std(),
 'miner_2_volscaled_reversal_1obs': -ret/ret.rolling(20,min_periods=15).std(),
 'miner_3_relative_volume_participation_20d': np.log(vol/vol.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
}
def sp(x,y):
 z=pd.concat([x,y],axis=1).dropna()
 return z.iloc[:,0].rank().corr(z.iloc[:,1].rank()) if len(z)>=8 else np.nan
def met(h):
 fw=panel.shift(-h)/panel-1
 ic=pd.Series({t:sp(fac.loc[t],fw.loc[t]) for t in panel.index}).dropna()
 turns=[1-sp(fac.iloc[i-1],fac.iloc[i]) for i in range(1,len(fac)) if pd.notna(sp(fac.iloc[i-1],fac.iloc[i]))]
 print(f'H={h} IC={ic.mean():.6f} ICIR={ic.mean()/ic.std(ddof=1):.6f} hit={(ic>0).mean():.4f} dates={len(ic)} mean_names={fac.notna().sum(1).mean():.2f} coverage={fac.notna().mean().mean():.4f} turnover={np.mean(turns):.6f}')
 for n,l,u in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31')]:
  x=ic.loc[l:u]; print(f' REGIME {n} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else f' REGIME {n} dates={len(x)}')
for h in (1,5,10,20): met(h)
print('FACTOR downside_asymmetry_resilience_20obs endpoint',panel.index.max().date(),'range',panel.index.min().date(),panel.index.max().date(),'assets',len(A))
mx=0; who=''; evidence=0
for n,x in lib.items():
 vals=[sp(fac.loc[t],x.loc[t]) for t in panel.index]; vals=[q for q in vals if pd.notna(q)]; m=max(map(abs,vals)) if vals else np.nan
 print('LIBCORR',n,'max_abs',round(m,6),'mean',round(np.mean(vals),6),'dates',len(vals))
 if pd.notna(m) and m>mx: mx,who=m,n
print('MAX_LIBRARY_CORRELATION',round(mx,6),who)
