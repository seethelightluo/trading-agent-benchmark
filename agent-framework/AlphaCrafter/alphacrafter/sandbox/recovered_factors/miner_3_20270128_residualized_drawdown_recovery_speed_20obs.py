"""miner_3: one idea: 20d recovery speed residualized cross-sectionally vs reversal and volatility."""
import json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan) if 'volume' in d else pd.Series(dtype=float)
panel=pd.DataFrame(P).sort_index(); ret=panel.pct_change(fill_method=None); vol=ret.rolling(20,min_periods=15).std(); net=panel.pct_change(20,fill_method=None); trend=net/vol.replace(0,np.nan); rev=-panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std().replace(0,np.nan)
raw=pd.DataFrame(index=panel.index,columns=A,dtype=float)
for a in A:
 s=panel[a]; low=s.rolling(20,min_periods=15).min(); age=s.rolling(20,min_periods=15).apply(lambda x:len(x)-1-np.argmin(x),raw=True); raw[a]=(s/low-1)/(age+1)
def residual(y,cs):
 out=pd.DataFrame(index=panel.index,columns=A,dtype=float)
 for dt in panel.index:
  z=pd.concat([y.loc[dt].rename('y')]+[x.loc[dt].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.column_stack([np.ones(len(z))]+[z[str(i)].values for i in range(len(cs))]); out.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
f=residual(raw,[rev,vol,trend])
acc=net-panel.shift(20).pct_change(40,fill_method=None); orth=residual(acc/vol.replace(0,np.nan),[trend]); rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}); spx=ret.SPX; bv=spx.rolling(20,min_periods=15).var().replace(0,np.nan); beta=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(spx)/bv for a in A})
d=get_index_daily_data('DXY',5000).set_index('date');d.index=pd.to_datetime(d.index);dr=pd.to_numeric(d.close,errors='coerce').pct_change();dv=dr.rolling(20,min_periods=15).var().replace(0,np.nan);dxy=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(dr)/dv.reindex(ret.index) for a in A})
v=get_index_daily_data('VIX',5000).set_index('date');v.index=pd.to_datetime(v.index);vr=pd.to_numeric(v.close,errors='coerce').pct_change();up=vr.where(vr>0);dn=vr.where(vr<0);vix=pd.DataFrame({a:ret[a].rolling(60,min_periods=50).cov(up)/up.rolling(60,min_periods=50).var()-ret[a].rolling(60,min_periods=50).cov(dn)/dn.rolling(60,min_periods=50).var() for a in A})
libs={'ravmom':trend,'risk_adjusted_trend':trend,'volnorm_reversal':rev,'realized_volatility':vol,'relative_volume':rv,'orthogonal_acceleration':orth,'negative_spx_beta':-beta,'dxy_beta':dxy,'vix_asymmetric_shock_beta':vix}
def stats(h):
 fw=panel.shift(-h)/panel-1; xs=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:xs.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(xs)); sd=x.std(ddof=1); turns=[]
 for i in range(1,len(f)):
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 reg={}
 for n,m in {'2020_2022':x.index.year<=2022,'2023_2024':x.index.year.isin([2023,2024]),'2025_2026':x.index.year>=2025}.items():
  q=x[m]; qsd=q.std(ddof=1);reg[n]={'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/qsd) if qsd else None,'hit_ratio':float((q>0).mean())}
 return {'horizon':h,'dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover':float(np.mean(turns)),'regimes':reg}
print('FACTOR residualized_drawdown_recovery_speed_20obs');print('DATA',panel.index.min().date(),panel.index.max().date(),'assets',len(A),'factor_cells',int(f.notna().sum().sum()),'of',f.size,'coverage',float(f.notna().mean().mean()))
for h in (1,5,10,20):print('METRIC',json.dumps(stats(h),sort_keys=True))
mx=0;who='';allc={}
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');allc[n]={'rho':float(rho),'cells':len(z)}
 if abs(rho)>mx:mx=abs(rho);who=n
print('LIBRARY',json.dumps(allc,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
