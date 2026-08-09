"""One-factor validation: 20-observation path-efficiency residual.
Signal is directional path efficiency (net 20d move divided by total absolute daily movement), cross-sectionally residualized from standardized 20d trend and realized volatility. It isolates orderly versus noisy movement rather than raw momentum level."""
import json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list'];P={};R={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).set_index('date');d.index=pd.to_datetime(d.index)
 P[a]=pd.to_numeric(d.close,errors='coerce').dropna();R[a]=P[a].pct_change(fill_method=None)
 V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan) if 'volume'in d else pd.Series(index=P[a].index,dtype=float)
panel=pd.DataFrame(P);ret=pd.DataFrame(R); vol=ret.rolling(20,min_periods=15).std()
net=panel.pct_change(20,fill_method=None); path=ret.abs().rolling(20,min_periods=15).sum(); eff=net/path.replace(0,np.nan)
trend=net/vol.replace(0,np.nan); f=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for dt in panel.index:
 z=pd.concat([eff.loc[dt].rename('e'),trend.loc[dt].rename('t'),vol.loc[dt].rename('v')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t,z.v]; f.loc[dt,z.index]=z.e-X@np.linalg.lstsq(X,z.e,rcond=None)[0]
# admitted-library reconstruction
acc=(net-panel.shift(20).pct_change(40,fill_method=None))/vol;orth=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for dt in panel.index:
 z=pd.concat([acc.loc[dt].rename('a'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.t];orth.loc[dt,z.index]=z.a-X@np.linalg.lstsq(X,z.a,rcond=None)[0]
rev=-panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std();rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A})
spx=ret.SPX;var=spx.rolling(20,min_periods=15).var().replace(0,np.nan); beta=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(spx)/var.reindex(ret[a].index) for a in A})
d=get_index_daily_data('DXY',5000).set_index('date');d.index=pd.to_datetime(d.index);dr=pd.to_numeric(d.close,errors='coerce').pct_change();dv=dr.rolling(20,min_periods=15).var().replace(0,np.nan);dxy=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(dr)/dv.reindex(ret[a].index) for a in A})
lib={'miner_1_ravmom_20obs':trend,'miner_3_risk_adjusted_trend_20d':trend,'miner_1_volnorm_reversal_5obs':rev,'miner_2_realized_volatility_20obs':vol,'miner_3_relative_volume_participation_20d':rv,'miner_3_orthogonal_trend_acceleration_20_60obs':orth,'miner_3_negative_spx_beta_20obs':-beta,'miner_2_dxy_beta_20obs':dxy}
def metric(h):
 fw=panel.shift(-h)/panel-1;q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 x=pd.Series(dict(q));turn=[]
 for i in range(1,len(f)):
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 regimes={};
 for name,mask in {'2020-2022':x.index.year<=2022,'2023-2024':x.index.year.isin([2023,2024]),'2025-2026':x.index.year>=2025}.items():
  y=x[mask];regimes[name]={'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std(ddof=1)),'hit_ratio':float((y>0).mean())}
 return {'horizon_days':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/x.std(ddof=1)),'ic_standard_error':float(x.std(ddof=1)/np.sqrt(len(x))),'ic_hit_ratio':float((x>0).mean()),'ic_dates':len(x),'mean_rank_turnover':float(np.mean(turn)),'mean_valid_coverage':float(np.mean([1 for _ in x])),'regimes':regimes}
print('FACTOR path_efficiency_residual_20obs; visible',panel.index.min().date(),panel.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',float(f.notna().mean().mean()),'mean_valid',float(f.notna().sum(axis=1).mean()))
for h in (1,5,10,20):print('METRIC',json.dumps(metric(h),sort_keys=True))
out={};mx=0
for n,x in lib.items():
 z=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');out[n]={'rho':float(rho),'common_signal_cells':len(z)};mx=max(mx,abs(rho))
print('LIBRARY_CORRELATION',json.dumps(out,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx)
