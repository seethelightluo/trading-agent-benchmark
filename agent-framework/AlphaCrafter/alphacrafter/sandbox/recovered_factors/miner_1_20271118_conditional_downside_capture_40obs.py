"""One-factor research: leave-one-out downside capture resilience, 40 observations.
Higher is better: standardized mean own return on days the cross-asset peer basket is down.
"""
import json, glob, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan) if 'volume' in d else pd.Series(index=d.index,dtype=float)
panel=pd.DataFrame(P).sort_index(); ret=panel.pct_change(fill_method=None)
vol=ret.rolling(20,min_periods=15).std().replace(0,np.nan)
# Leave-one-out conditioning avoids own return determining whether a day is classified risk-off.
f=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for a in A:
 peer=ret.drop(columns=a).mean(axis=1)
 f[a]=ret[a].where(peer<0).rolling(40,min_periods=12).mean()/vol[a]
# Reconstruct all admitted signals for mandatory correlation test.
def residual(y,cs):
 o=pd.DataFrame(index=panel.index,columns=A,dtype=float)
 for dt in panel.index:
  z=pd.concat([y.loc[dt].rename('y')]+[c.loc[dt].rename(str(i)) for i,c in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.column_stack([np.ones(len(z))]+[z[str(i)].values for i in range(len(cs))]);o.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
 return o
net=panel.pct_change(20,fill_method=None); trend=net/vol; acc=net-panel.shift(20).pct_change(40,fill_method=None); orth=residual(acc/vol,[trend]); rev=-panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std().replace(0,np.nan)
rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A})
spx=ret.SPX; bv=spx.rolling(20,min_periods=15).var().replace(0,np.nan); beta=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(spx)/bv for a in A})
d=get_index_daily_data('DXY',5000).set_index('date');d.index=pd.to_datetime(d.index);dr=pd.to_numeric(d.close,errors='coerce').pct_change();dv=dr.rolling(20,min_periods=15).var().replace(0,np.nan);dxy=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(dr)/dv.reindex(ret.index) for a in A})
v=get_index_daily_data('VIX',5000).set_index('date');v.index=pd.to_datetime(v.index);vr=pd.to_numeric(v.close,errors='coerce').pct_change();up=vr.where(vr>0);dn=vr.where(vr<0);vix=pd.DataFrame({a:ret[a].rolling(60,min_periods=25).cov(up)/up.rolling(60,min_periods=25).var()-ret[a].rolling(60,min_periods=25).cov(dn)/dn.rolling(60,min_periods=25).var() for a in A})
kurt=-ret.rolling(40,min_periods=30).kurt()
def es(x): q=x.quantile(.2); return x[x<=q].mean()
ies=-ret.rolling(40,min_periods=30).apply(es,raw=False)/vol
# Existing miner_1 downside peer correlation and return autocorrelation.
dpc=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for a in A:
 peer=ret.drop(columns=a).mean(axis=1); down=peer.where(peer<0)
 dpc[a]=-(ret[a].rolling(40,min_periods=12).cov(down)/down.rolling(40,min_periods=12).std()/ret[a].rolling(40,min_periods=12).std())
auto=ret.rolling(20,min_periods=16).corr(ret.shift(1))
libs={'ravmom_20obs':trend,'risk_adjusted_trend_20d':trend,'volnorm_reversal_5obs':rev,'relative_volume_participation_20d':rv,'orthogonal_trend_acceleration_20_60obs':orth,'negative_spx_beta_20obs':-beta,'downside_peer_correlation_40obs':dpc,'inverse_excess_kurtosis_40obs':kurt,'inverse_expected_shortfall_40obs':ies,'inverse_upside_peer_correlation_40obs':None,'negative_conditional_dxy_up_beta_40obs':None,'positive_conditional_dxy_down_beta_40obs':None,'asymmetric_peer_beta_resilience_40obs':None,'return_autocorrelation_20obs':auto}
# Read persisted expression signals unavailable only for three conditional factors: reconstruct.
uppeer=pd.DataFrame(index=panel.index,columns=A,dtype=float); dxyup=pd.DataFrame(index=panel.index,columns=A,dtype=float); dxydn=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for a in A:
 peer=ret.drop(columns=a).mean(axis=1); u=peer.where(peer>0); uppeer[a]=-(ret[a].rolling(40,min_periods=12).cov(u)/u.rolling(40,min_periods=12).std()/ret[a].rolling(40,min_periods=12).std())
 du=dr.where(dr>0); dd=dr.where(dr<0); dxyup[a]=-(ret[a].rolling(40,min_periods=12).cov(du)/du.rolling(40,min_periods=12).var()); dxydn[a]=ret[a].rolling(40,min_periods=12).cov(dd)/dd.rolling(40,min_periods=12).var()
libs['inverse_upside_peer_correlation_40obs']=uppeer;libs['negative_conditional_dxy_up_beta_40obs']=dxyup;libs['positive_conditional_dxy_down_beta_40obs']=dxydn
# asymmetric peer beta: negative down beta + positive up beta
asym=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for a in A:
 peer=ret.drop(columns=a).mean(axis=1); u=peer.where(peer>0); dn0=peer.where(peer<0)
 asym[a]=ret[a].rolling(40,min_periods=12).cov(u)/u.rolling(40,min_periods=12).var()-ret[a].rolling(40,min_periods=12).cov(dn0)/dn0.rolling(40,min_periods=12).var()
libs['asymmetric_peer_beta_resilience_40obs']=asym
def stats(h):
 fw=panel.shift(-h)/panel-1; rows=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows)); sd=x.std(ddof=1); turn=[]
 for i in range(1,len(f)):
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 reg={}
 for n,m in {'2020_2022':x.index.year<=2022,'2023_2024':x.index.year.isin([2023,2024]),'2025_2026':x.index.year.isin([2025,2026]),'2027':x.index.year==2027}.items():
  q=x[m];reg[n]={'dates':len(q),'ic':q.mean() if len(q) else None,'icir':q.mean()/q.std(ddof=1) if len(q)>1 and q.std(ddof=1)>0 else None,'hit_ratio':(q>0).mean() if len(q) else None}
 return {'horizon':h,'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_instruments':np.mean(ns),'ic_se':sd/np.sqrt(len(x)),'mean_rank_turnover':np.mean(turn),'regimes':reg}
print('DATA',panel.index.min().date(),panel.index.max().date(),'assets',len(A),'factor_cells',int(f.notna().sum().sum()),'total_cells',f.size,'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:print('METRIC',json.dumps(stats(h),default=float,sort_keys=True))
out={};mx=-1;who=None
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');out[n]={'rho':rho,'cells':len(z)}
 if abs(rho)>mx:mx=abs(rho);who=n
print('LIBRARY',json.dumps(out,default=float,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who,'LIBRARY_FILES',len([x for x in glob.glob('factors/*.json')]))
