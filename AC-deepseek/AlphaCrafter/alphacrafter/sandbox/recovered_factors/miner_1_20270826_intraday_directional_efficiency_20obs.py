"""miner_1: intraday directional efficiency, trailing 20 completed observations."""
import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-08-25')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END] for a in A}
p=pd.DataFrame({a:D[a].close.astype(float) for a in A}); o=pd.DataFrame({a:D[a].open.astype(float) for a in A}); v=pd.DataFrame({a:D[a].volume.astype(float).replace(0,np.nan) for a in A}); r=p.pct_change(); intra=p/o-1; med=r.median(axis=1);disp=r.std(axis=1)
# The signed efficiency of open-to-close returns: persistent intraday buying/selling rather than noisy sessions.
f=intra.rolling(20,min_periods=15).sum()/intra.abs().rolling(20,min_periods=15).sum()
def rollsp(x,y,w):
 out=[]
 for t in x.index:
  q=pd.concat([x,y],axis=1).loc[:t].tail(w).dropna();out.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman') if len(q)>=max(15,w*3//4) else np.nan)
 return pd.Series(out,index=x.index)
vol=r.rolling(20,min_periods=15).std();fast=(p/p.shift(20)-1)/vol;slow=(p/p.shift(60)-1)/r.rolling(60,min_periods=45).std()
lib={'risk_adjusted_trend_20d':fast,'relative_volume_participation_20d':np.log(v/v.rolling(20,min_periods=15).mean()),'realized_volatility_20obs':vol,'volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'risk_adjusted_trend_acceleration_20_60d':fast-slow,'return_persistence_autocorr_20obs':r.rolling(20,min_periods=15).corr(r.shift(1)),'return_sign_balance_20obs':(r>0).rolling(20,min_periods=15).mean(),'return_directional_efficiency_20obs':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),'relative_liquidity_stress_20_60obs':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=45).mean()),'dispersion_sensitivity_20obs':pd.DataFrame({a:rollsp(r[a],disp,20) for a in A}),'volatility_clustering_autocorr_20obs':r.abs().rolling(20,min_periods=15).apply(lambda x:x.autocorr(1),raw=False)}
ca={};ds={};bt={};ex={}
for a in A:
 aa=[];dd=[];bb=[];ee=[]
 for t in r.index:
  q=pd.concat([r[a],med],axis=1).loc[:t].tail(60).dropna();dn=q[q.iloc[:,1]<0];up=q[q.iloc[:,1]>=0]
  if len(q)>=45 and q.iloc[:,1].var()>0:
   b=q.iloc[:,0].cov(q.iloc[:,1])/q.iloc[:,1].var();res=q.iloc[:,0]-(q.iloc[:,0].mean()+b*(q.iloc[:,1]-q.iloc[:,1].mean()));dd.append(np.sqrt(np.mean(np.minimum(res,0)**2))/np.sqrt(np.mean(res**2)));bb.append(b)
  else: dd.append(np.nan);bb.append(np.nan)
  aa.append(dn.iloc[:,0].corr(dn.iloc[:,1])-up.iloc[:,0].corr(up.iloc[:,1]) if len(dn)>=10 and len(up)>=10 else np.nan)
  ee.append(dn.iloc[:,0].cov(dn.iloc[:,1])/dn.iloc[:,1].var()-bb[-1] if len(dn)>=12 and dn.iloc[:,1].var()>0 and np.isfinite(bb[-1]) else np.nan)
 ca[a]=aa;ds[a]=dd;bt[a]=bb;ex[a]=ee
ca=pd.DataFrame(ca,index=r.index);ds=pd.DataFrame(ds,index=r.index);bt=pd.DataFrame(bt,index=r.index);ex=pd.DataFrame(ex,index=r.index)
def residual(x,z):
 q=pd.DataFrame(np.nan,index=x.index,columns=A)
 for t in x.index:
  w=pd.concat([x.loc[t],z.loc[t]],axis=1).dropna()
  if len(w)>=8 and w.iloc[:,1].var()>0:q.loc[t,w.index]=w.iloc[:,0]-np.polyval(np.polyfit(w.iloc[:,1],w.iloc[:,0],1),w.iloc[:,1])
 return q
lib.update({'correlation_asymmetry_60obs':ca,'residual_downside_semivol_share_60obs':pd.DataFrame(ds,index=r.index),'vol_orthogonal_median_beta_60obs':residual(pd.DataFrame(bt,index=r.index),vol),'excess_downside_beta_ca_orthogonal_60obs':residual(pd.DataFrame(ex,index=r.index),ca)})
print('FACTOR intraday_directional_efficiency_20obs: sum(close/open-1) / sum(abs(close/open-1)), trailing 20 sessions')
print('visible_through',END.date(),'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
best=None; results={}
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1;ics=[];coverage=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna();coverage.append(len(q)/15)
  if len(q)>=8:ics.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
 x=pd.Series([z[1] for z in ics],index=pd.DatetimeIndex([z[0] for z in ics]));ir=x.mean()/x.std(ddof=1);results[h]=(x,ir,np.mean(coverage));print(f'H={h} dates={len(x)} IC={x.mean():.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} coverage={np.mean(coverage):.4f} mean_assets={15*np.mean(coverage):.2f}')
 if best is None or abs(x.mean())*abs(ir)>best[0]:best=(abs(x.mean())*abs(ir),h,x)
x=best[2];print('selected_decay_horizon',best[1])
for n,mask in [('2020',x.index<'2021'),('2021-22',(x.index>='2021')&(x.index<'2023')),('2023-24',(x.index>='2023')&(x.index<'2025')),('2025-current',x.index>='2025')]:
 q=x[mask];print(f'REGIME {n} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('rank_turnover=%.6f'%np.mean(to))
mx=-1
for n,z in lib.items():
 q=pd.concat([f.stack(),z.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman');print(f'LIB {n} rho={rho:.6f} cells={len(q)}')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print(f'max_abs_library_correlation={mx:.6f}; closest={who}; evidence_cells={cells}; library_count={len(lib)}')
