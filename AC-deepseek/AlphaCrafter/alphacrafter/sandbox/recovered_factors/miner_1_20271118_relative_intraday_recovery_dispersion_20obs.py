"""One idea: relative intraday recovery dispersion (20 observations), visible through prior day."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2027-11-17'
def load(a,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().loc[:END]
d={a:load(a) for a in A};c=pd.DataFrame({a:x.close for a,x in d.items()});o=pd.DataFrame({a:x.open for a,x in d.items()});v=pd.DataFrame({a:x.volume.replace(0,np.nan) for a,x in d.items()});r=c.pct_change();med=r.median(axis=1)
# High value: asset's intraday recovery variability is high relative to its total daily variability.
intra=c/o-1
f=intra.rolling(20,min_periods=15).std()/r.rolling(20,min_periods=15).std()
ret20=r.rolling(20,min_periods=15).sum(); vol20=r.rolling(20,min_periods=15).std();ret60=r.rolling(60,min_periods=45).sum();vol60=r.rolling(60,min_periods=45).std()
L={'risk_adjusted_trend_20d':ret20/vol20,'relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'realized_volatility_20obs':vol20,'volnorm_reversal_5obs':-r.rolling(5,min_periods=4).sum()/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':ret20/vol20-ret60/vol60,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'return_directional_efficiency_20obs':ret20.abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean()),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'dispersion_sensitivity_20obs':r.rolling(20,min_periods=15).corr(r.std(axis=1)),'downside_volume_participation_asymmetry_60obs':np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean())}
ca=pd.DataFrame(index=r.index,columns=A,dtype=float);xdb=ca.copy()
for i,t in enumerate(r.index):
 z=r.iloc[max(0,i-59):i+1]; down=med.loc[z.index]<0
 for a in A:
  q=pd.concat([z[a],med.loc[z.index]],axis=1);lo=q[down].dropna();hi=q[~down].dropna()
  if len(lo)>=8 and len(hi)>=8:ca.loc[t,a]=lo.iloc[:,0].corr(lo.iloc[:,1])-hi.iloc[:,0].corr(hi.iloc[:,1])
  if len(lo)>=8 and lo.iloc[:,1].var()>0:xdb.loc[t,a]=lo.iloc[:,0].cov(lo.iloc[:,1])/lo.iloc[:,1].var()
L['correlation_asymmetry_60obs']=ca
beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0);resid=r-beta.mul(med,axis=0)
L['residual_downside_semivol_share_60obs']=np.sqrt(resid.where(resid<0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(resid.pow(2).rolling(60,min_periods=45).mean())
ort=pd.DataFrame(index=r.index,columns=A)
for t in r.index:
 q=pd.concat([beta.loc[t],vol20.loc[t]],axis=1).dropna()
 if len(q)>=3:ort.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
L['vol_orthogonal_median_beta_60obs']=ort
ex=pd.DataFrame(index=r.index,columns=A)
for t in r.index:
 q=pd.concat([xdb.loc[t],ca.loc[t]],axis=1).dropna()
 if len(q)>=3:ex.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
L['excess_downside_beta_ca_orthogonal_60obs']=ex
vx=load('VIX',True).close.pct_change().reindex(r.index);m=vx.where(vx<0);mask=vx<0;bb={}
for w in [25,60]:
 n=mask.astype(float).rolling(w,min_periods=12).sum();sx=m.rolling(w,min_periods=12).sum();den=(m*m).rolling(w,min_periods=12).sum()-sx*sx/n
 bb[w]=pd.DataFrame({a:((m*r[a].where(mask)).rolling(w,min_periods=12).sum()-sx*r[a].where(mask).rolling(w,min_periods=12).sum()/n)/den for a in A})
L['adaptive_vix_relief_beta_change_25_60obs']=bb[25]-bb[60]
dates=f.index[f.notna().sum(axis=1)>=8];print('FACTOR relative_intraday_recovery_dispersion_20obs visible_through',END,'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
allh={}
for h in [1,5,10,20]:
 y=c.shift(-h)/c-1;z=[];nn=[]
 for t in dates:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:z.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));nn.append(len(q))
 s=pd.Series(dict(z));allh[h]=s;print('H',h,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(nn)))
bh=max(allh,key=lambda h:abs(allh[h].mean()*allh[h].mean()/allh[h].std(ddof=1)));s=allh[bh];print('SELECTED',bh)
for lo,hi,nm in [('2020','2021','2020'),('2021','2023','2021-22'),('2023','2025','2023-24'),('2025','2030','2025-current')]:
 q=s[(s.index>=lo)&(s.index<hi)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan,'hit',round((q>0).mean(),4) if len(q) else np.nan)
ranks=f.rank(axis=1);to=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i],ranks.iloc[i-1]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER',round(np.mean(to),6))
mx=(-1,None,0)
for name,z in L.items():
 vals=[]
 for t in dates:
  q=pd.concat([f.loc[t],z.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   w=abs(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
   if np.isfinite(w):vals.append(w)
 k=max(vals) if vals else np.nan;print('LIB',name,'max_abs_rho',k,'dates',len(vals))
 if np.isfinite(k) and k>mx[0]:mx=(k,name,len(vals))
print('MAX_ABS_LIBRARY_CORRELATION %.6f closest %s dates %d'%mx)
