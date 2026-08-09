"""One idea: downside range-volume confirmation correlation over 20 observations.
High values indicate that an asset's unusually wide intraday ranges coincide with
unusually high volume specifically on its down-return observations. This probes
stress-participation linkage, not raw return trend, volume level, or volatility.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2028-08-23'
def load(a,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
d={a:load(a) for a in A}; c=pd.DataFrame({a:x.close for a,x in d.items()}); v=pd.DataFrame({a:x.volume.replace(0,np.nan) for a,x in d.items()}); r=c.pct_change(); med=r.median(axis=1)
o=pd.DataFrame({a:x.open for a,x in d.items()}); hi=pd.DataFrame({a:x.high for a,x in d.items()}); lo=pd.DataFrame({a:x.low for a,x in d.items()})
# Candidate: 20-observation correlation, restricted to own down days, of log intraday range
# and log relative volume. Minimum 8 adverse observations prevents a mechanical sparse signal.
rng=np.log(((hi-lo)/c.shift(1)).replace(0,np.nan)); rv=np.log(v/v.rolling(20,min_periods=15).mean())
f=rng.where(r<0).rolling(20,min_periods=8).corr(rv.where(r<0))
# Current admitted-library signal reconstructions.
vol20=r.rolling(20,min_periods=15).std(); ret20=r.rolling(20,min_periods=15).sum();ret60=r.rolling(60,min_periods=45).sum();vol60=r.rolling(60,min_periods=45).std()
L={'risk_adjusted_trend_20d':ret20/vol20,'relative_volume_participation_20d':rv,'realized_volatility_20obs':vol20,'volnorm_reversal_5obs':-r.rolling(5,min_periods=4).sum()/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':ret20/vol20-ret60/vol60,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'return_directional_efficiency_20obs':ret20.abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean()),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'dispersion_sensitivity_20obs':r.rolling(20,min_periods=15).corr(r.std(axis=1)),'downside_volume_participation_asymmetry_60obs':np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean())}
ca=pd.DataFrame(index=r.index,columns=A,dtype=float);xdb=ca.copy()
for t in r.index:
 z=r.loc[:t].tail(60); down=med.loc[z.index]<0
 for a in A:
  q=pd.concat([z[a],med.loc[z.index]],axis=1);dn=q[down].dropna();up=q[~down].dropna()
  if len(dn)>=8 and len(up)>=8:ca.loc[t,a]=dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1])
  if len(dn)>=8 and dn.iloc[:,1].var()!=0:xdb.loc[t,a]=dn.iloc[:,0].cov(dn.iloc[:,1])/dn.iloc[:,1].var()
L['correlation_asymmetry_60obs']=ca; beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0);resid=r-beta.mul(med,axis=0)
L['residual_downside_semivol_share_60obs']=np.sqrt(resid.where(resid<0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(resid.pow(2).rolling(60,min_periods=45).mean())
ort=pd.DataFrame(index=r.index,columns=A,dtype=float);ex=ort.copy()
for t in r.index:
 q=pd.concat([beta.loc[t],vol20.loc[t]],axis=1).dropna()
 if len(q)>=3:ort.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 q=pd.concat([xdb.loc[t],ca.loc[t]],axis=1).dropna()
 if len(q)>=3:ex.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
L['vol_orthogonal_median_beta_60obs']=ort;L['excess_downside_beta_ca_orthogonal_60obs']=ex
vx=load('VIX',True).close.pct_change().reindex(r.index)
def cb(mask,w=60):
 x=vx.where(mask); n=mask.astype(float).rolling(w,min_periods=12).sum();sx=x.rolling(w,min_periods=12).sum();den=(x*x).rolling(w,min_periods=12).sum()-sx*sx/n
 return pd.DataFrame({a:((x*r[a].where(mask)).rolling(w,min_periods=12).sum()-sx*r[a].where(mask).rolling(w,min_periods=12).sum()/n)/den for a in A})
L['vix_shock_relief_beta_asymmetry_60obs']=cb(vx>0)-cb(vx<0);L['adaptive_vix_relief_beta_change_25_60obs']=cb(vx<0,25)-cb(vx<0,60)
dv20=np.log(v.where(r<0).rolling(20,min_periods=10).mean()/v.where(r>=0).rolling(20,min_periods=10).mean());dv60=np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean())
L['inverted_downside_volume_participation_acceleration_20_60obs']=-(dv20-dv60);L['overnight_daytime_reversal_concordance_20obs']=(-((o/c.shift(1)-1)*(c/o-1))).rolling(20,min_periods=15).mean()
# Macro-conditioned factors currently admitted since the preceding reconstruction.
dxy=load('DXY',True).close.pct_change().reindex(r.index)
def dbeta(x,mask,w=60,minp=12):
 x=x.where(mask);n=mask.astype(float).rolling(w,min_periods=minp).sum();sx=x.rolling(w,min_periods=minp).sum();den=(x*x).rolling(w,min_periods=minp).sum()-sx*sx/n
 return pd.DataFrame({a:((x*r[a].where(mask)).rolling(w,min_periods=minp).sum()-sx*r[a].where(mask).rolling(w,min_periods=minp).sum()/n)/den for a in A})
dvol=abs(dxy).rolling(60,min_periods=45).median(); dm=dxy.rolling(20,min_periods=15).sum()
L['dxy_median_trend_regime_beta_spread_60obs']=dbeta(dxy,dm>dm.rolling(60,min_periods=45).median())-dbeta(dxy,dm<=dm.rolling(60,min_periods=45).median())
L['dxy_relative_vol_regime_beta_spread_60obs']=dbeta(dxy,abs(dxy)>dvol)-dbeta(dxy,abs(dxy)<=dvol)
eval_dates=f.index[f.notna().sum(axis=1)>=8]; allh={}
print('FACTOR downside_range_volume_confirmation_correlation_20obs visible_through',END,'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
for h in [1,5,10,20]:
 vals=[];ns=[];y=c.shift(-h)/c-1
 for t in eval_dates:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:vals.append((t,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic));ns.append(len(q))
 s=pd.Series(dict(vals));allh[h]=s;print('H',h,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(ns)))
bh=max(allh,key=lambda h:abs(allh[h].mean()*allh[h].mean()/allh[h].std(ddof=1)));s=allh[bh];print('SELECTED',bh)
for lo,hi,nm in [('2020','2021','2020'),('2021','2023','2021-22'),('2023','2025','2023-24'),('2025','2030','2025-current')]:
 q=s[(s.index>=lo)&(s.index<hi)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
ranks=f.rank(axis=1);to=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i],ranks.iloc[i-1]],axis=1).dropna()
 if len(q)>=8:to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(np.mean(to),6),'coverage',round(f.notna().mean().mean(),4))
mx=(-1,None,0); evidence=0
for name,g in L.items():
 vals=[]
 for t in eval_dates:
  q=pd.concat([f.loc[t],g.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:vals.append(abs(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 k=max(vals) if vals else np.nan;print('LIB',name,'max_abs_rho',k,'dates',len(vals));evidence+=len(vals)
 if np.isfinite(k) and k>mx[0]:mx=(k,name,len(vals))
print('MAX_ABS_LIBRARY_CORRELATION %.6f closest %s dates %d evidence_cells %d'%(*mx,evidence))
