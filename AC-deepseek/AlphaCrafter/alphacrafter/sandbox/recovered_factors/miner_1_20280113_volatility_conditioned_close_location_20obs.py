"""One-factor validation: volatility-conditioned close-location persistence."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2028-01-12'
def load(a,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().loc[:END]
d={a:load(a) for a in A}; c=pd.DataFrame({a:x.close for a,x in d.items()}); o=pd.DataFrame({a:x.open for a,x in d.items()}); hi=pd.DataFrame({a:x.high for a,x in d.items()}); lo=pd.DataFrame({a:x.low for a,x in d.items()}); v=pd.DataFrame({a:x.volume.replace(0,np.nan) for a,x in d.items()}); r=c.pct_change(); med=r.median(1)
# Continuous close-location is amplified only when yesterday's own 20d volatility is elevated cross-sectionally.
# This asks whether sustained ability to close near the daily high in a volatile asset predicts subsequent relative return.
clv=((c-lo)/(hi-lo).replace(0,np.nan)*2-1).clip(-1,1); vol20=r.rolling(20,min_periods=15).std(); vpct=vol20.rank(axis=1,pct=True).shift(1)
f=(clv*vpct).rolling(20,min_periods=15).mean()
ret20=r.rolling(20,min_periods=15).sum(); ret60=r.rolling(60,min_periods=45).sum(); vol60=r.rolling(60,min_periods=45).std(); rv=np.log(v/v.rolling(20,min_periods=15).mean())
L={'risk_adjusted_trend_20d':ret20/vol20,'relative_volume_participation_20d':rv,'realized_volatility_20obs':vol20,'volnorm_reversal_5obs':-r.rolling(5,min_periods=4).sum()/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':ret20/vol20-ret60/vol60,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'return_directional_efficiency_20obs':ret20.abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean()),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean()-.5,'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1)),'dispersion_sensitivity_20obs':r.rolling(20,min_periods=15).corr(r.std(axis=1)),'downside_volume_participation_asymmetry_60obs':np.log(v.where(r<0).rolling(60,min_periods=30).mean()/v.where(r>=0).rolling(60,min_periods=30).mean())}
# downside participation acceleration
short=np.log(v.where(r<0).rolling(20,min_periods=5).mean()/v.where(r>=0).rolling(20,min_periods=5).mean()); long=np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean()); L['inverted_downside_volume_participation_acceleration_20_60obs']=long-short
ca=pd.DataFrame(index=r.index,columns=A,dtype=float);xdb=ca.copy()
for t in r.index:
 z=r.loc[:t].tail(60); down=med.loc[z.index]<0
 for a in A:
  x=pd.concat([z[a],med.loc[z.index]],axis=1);q=x[down].dropna();w=x[~down].dropna()
  if len(q)>=8 and len(w)>=8:ca.loc[t,a]=q.iloc[:,0].corr(q.iloc[:,1])-w.iloc[:,0].corr(w.iloc[:,1])
  if len(q)>=8 and q.iloc[:,1].var()!=0:xdb.loc[t,a]=q.iloc[:,0].cov(q.iloc[:,1])/q.iloc[:,1].var()
L['correlation_asymmetry_60obs']=ca
beta=r.rolling(60,min_periods=45).cov(med).div(med.rolling(60,min_periods=45).var(),axis=0);resid=r-beta.mul(med,axis=0)
L['residual_downside_semivol_share_60obs']=np.sqrt(resid.where(resid<0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(resid.pow(2).rolling(60,min_periods=45).mean())
ort=pd.DataFrame(index=r.index,columns=A,dtype=float);ex=ort.copy()
for t in r.index:
 q=pd.concat([beta.loc[t],vol20.loc[t]],axis=1).dropna()
 if len(q)>=3:ort.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
 q=pd.concat([xdb.loc[t],ca.loc[t]],axis=1).dropna()
 if len(q)>=3:ex.loc[t,q.index]=q.iloc[:,0]-np.polyval(np.polyfit(q.iloc[:,1],q.iloc[:,0],1),q.iloc[:,1])
L['vol_orthogonal_median_beta_60obs']=ort;L['excess_downside_beta_ca_orthogonal_60obs']=ex
vx=load('VIX',True).close.pct_change().reindex(r.index);mask=vx<0;x=vx.where(mask);bb={}
for w in [25,60]:
 n=mask.astype(float).rolling(w,min_periods=12).sum();sx=x.rolling(w,min_periods=12).sum();den=(x*x).rolling(w,min_periods=12).sum()-sx*sx/n
 bb[w]=pd.DataFrame({a:((x*r[a].where(mask)).rolling(w,min_periods=12).sum()-sx*r[a].where(mask).rolling(w,min_periods=12).sum()/n)/den for a in A})
L['adaptive_vix_relief_beta_change_25_60obs']=bb[25]-bb[60]
eval_dates=f.index[f.notna().sum(1)>=8]
def getic(y):
 z=[];ns=[]
 for t in eval_dates:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append((t,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic));ns.append(len(q))
 return pd.Series(dict(z)),np.mean(ns)
print('FACTOR volatility_conditioned_close_location_20obs visible_through',END,'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
H={}
for h in [1,5,10,20]:
 s,n=getic(c.shift(-h)/c-1);H[h]=s;print('H',h,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),n))
bh=max(H,key=lambda h:abs(H[h].mean()*H[h].mean()/H[h].std(ddof=1));s=H[bh];print('SELECTED',bh)
for l,u,nm in [('2020','2021','2020'),('2021','2023','2021-22'),('2023','2025','2023-24'),('2025','2030','2025-current')]:
 q=s[(s.index>=l)&(s.index<u)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rk=f.rank(1);turn=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i],rk.iloc[i-1]],axis=1).dropna()
 if len(q)>=8:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(np.mean(turn),6),'coverage',round(f.notna().mean().mean(),4))
mx=(-1,None,0);evidence=0
for name,g in L.items():
 vals=[]
 for t in eval_dates:
  q=pd.concat([f.loc[t],g.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:vals.append(abs(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
 k=max(vals) if vals else np.nan;print('LIB',name,'max_abs_rho',k,'dates',len(vals));evidence+=len(vals)
 if np.isfinite(k) and k>mx[0]:mx=(k,name,len(vals))
print('MAX_ABS_LIBRARY_CORRELATION %.6f closest %s dates %d evidence_cells %d'%(*mx,evidence))
