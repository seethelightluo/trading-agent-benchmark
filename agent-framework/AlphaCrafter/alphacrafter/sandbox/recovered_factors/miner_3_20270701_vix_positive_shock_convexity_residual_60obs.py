"""One-factor validation: VIX positive-shock convexity residual (60 observations)."""
import json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan) if 'volume' in d else pd.Series(dtype=float)
panel=pd.DataFrame(P).sort_index(); ret=panel.pct_change(fill_method=None)
def ix(sym):
 d=get_index_daily_data(sym,5000).set_index('date');d.index=pd.to_datetime(d.index);return pd.to_numeric(d.close,errors='coerce').sort_index().pct_change()
spx=ret.SPX; vr=ix('VIX').reindex(panel.index); z=vr/vr.rolling(60,min_periods=35).std(); q=(z.where(z>0)**2)
# per-asset rolling OLS: coefficient on squared positive standardized VIX shock after linear VIX shock.
def convex(y):
 out=pd.Series(index=panel.index,dtype=float)
 for i in range(59,len(panel)):
  d=pd.concat([y.iloc[i-59:i+1],z.iloc[i-59:i+1],q.iloc[i-59:i+1]],axis=1).dropna()
  if len(d)>=35:
   X=np.column_stack([np.ones(len(d)),d.iloc[:,1],d.iloc[:,2]])
   out.iloc[i]=np.linalg.lstsq(X,d.iloc[:,0].values,rcond=None)[0][2]
 return out
raw=pd.DataFrame({a:convex(ret[a]) for a in A})
# remove contemporaneous linear VIX-asymmetry cross-sectionally: focus on incremental convexity.
up=vr.where(vr>0);dn=vr.where(vr<0)
vixlin=pd.DataFrame({a:ret[a].rolling(60,min_periods=35).cov(up)/up.rolling(60,min_periods=35).var()-ret[a].rolling(60,min_periods=35).cov(dn)/dn.rolling(60,min_periods=35).var() for a in A})
def residual(y,x):
 o=pd.DataFrame(index=y.index,columns=A,dtype=float)
 for dt in y.index:
  d=pd.concat([y.loc[dt].rename('y'),x.loc[dt].rename('x')],axis=1).dropna()
  if len(d)>=8:
   X=np.column_stack([np.ones(len(d)),d.x]);o.loc[dt,d.index]=d.y-X@np.linalg.lstsq(X,d.y,rcond=None)[0]
 return o
f=residual(raw,vixlin)
v20=ret.rolling(20,min_periods=15).std(); trend=panel.pct_change(20,fill_method=None)/v20.replace(0,np.nan); rev=-panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std().replace(0,np.nan)
acc=panel.pct_change(20,fill_method=None)-panel.shift(20).pct_change(40,fill_method=None);orth=residual(acc/v20.replace(0,np.nan),trend)
rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}); bv=spx.rolling(20,min_periods=15).var(); spxb=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(spx)/bv for a in A})
dr=ix('DXY').reindex(panel.index);db=dr.rolling(20,min_periods=15).var();dxy=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(dr)/db for a in A})
kurt=-ret.rolling(40,min_periods=30).kurt(); tail=ret.rolling(40,min_periods=30).apply(lambda x:x[x<=x.quantile(.2)].mean(),raw=False);ies=-tail/v20
libs={'risk_adjusted_trend':trend,'relative_volume_participation':rv,'realized_volatility':v20,'ravmom':trend,'volnorm_reversal':rev,'orthogonal_trend_acceleration':orth,'negative_spx_beta':-spxb,'dxy_beta':dxy,'vix_asymmetric_shock_beta':vixlin,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':ies}
def metrics(h):
 fw=panel.shift(-h)/panel-1; rr=[]; nn=[]
 for dt in f.index:
  d=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(d)>=8:rr.append((dt,d.f.corr(d.r,method='spearman')));nn.append(len(d))
 x=pd.Series(dict(rr));sd=x.std(ddof=1); turns=[]
 for i in range(1,len(f)):
  d=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(d)>=8:turns.append(1-d.iloc[:,0].corr(d.iloc[:,1],method='spearman'))
 regs={}
 for name,mask in {'2020_2022':x.index.year<=2022,'2023_2024':x.index.year.isin([2023,2024]),'2025_2026':x.index.year.isin([2025,2026]),'2027':x.index.year==2027}.items():
  a=x[mask]; regs[name]={'dates':len(a),'ic':float(a.mean()) if len(a) else None,'icir':float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else None,'hit_ratio':float((a>0).mean()) if len(a) else None}
 return {'horizon':h,'dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(nn)),'mean_rank_turnover':float(np.mean(turns)),'regimes':regs}
print('DATA',panel.index.min().date(),panel.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'of',f.size,'coverage',float(f.notna().mean().mean()))
for h in [1,5,10,20]:print('METRIC',json.dumps(metrics(h),sort_keys=True))
cc={};mx=-1;who=None
for n,x in libs.items():
 d=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();r=d.f.corr(d.x,method='spearman') if len(d)>2 else np.nan;cc[n]={'rho':None if pd.isna(r) else float(r),'cells':len(d)}
 if not pd.isna(r) and abs(r)>mx:mx=abs(r);who=n
print('LIBRARY',json.dumps(cc,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
