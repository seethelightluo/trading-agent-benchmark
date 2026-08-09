"""One-factor validation: inverse standardized expected-shortfall severity (40 observations).
For each asset, calculate the mean of daily returns at/below its own trailing 20th-percentile,
divided by trailing volatility, then negate. Higher values identify relatively resilient left tails.
"""
import json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; P={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan) if 'volume' in d else pd.Series(index=d.index,dtype=float)
panel=pd.DataFrame(P).sort_index(); ret=panel.pct_change(fill_method=None)
# Rolling expected-shortfall approximation: mean of the worst 20% own observations, standardized by volatility.
def es(x):
 q=x.quantile(.20); return x[x<=q].mean()
tail=ret.rolling(40,min_periods=30).apply(es,raw=False); vol=ret.rolling(20,min_periods=15).std()
f=-tail/vol.replace(0,np.nan)
net=panel.pct_change(20,fill_method=None);trend=net/vol.replace(0,np.nan)
def residual(y,cs):
 o=pd.DataFrame(index=panel.index,columns=A,dtype=float)
 for dt in panel.index:
  z=pd.concat([y.loc[dt].rename('y')]+[c.loc[dt].rename(str(i)) for i,c in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.column_stack([np.ones(len(z))]+[z[str(i)].values for i in range(len(cs))]); o.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y.values,rcond=None)[0]
 return o
acc=net-panel.shift(20).pct_change(40,fill_method=None);orth=residual(acc/vol.replace(0,np.nan),[trend]); rev=-panel.pct_change(5,fill_method=None)/ret.rolling(5,min_periods=4).std().replace(0,np.nan)
rv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}); spx=ret.SPX;bvar=spx.rolling(20,min_periods=15).var().replace(0,np.nan);beta=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(spx)/bvar for a in A})
d=get_index_daily_data('DXY',5000).set_index('date');d.index=pd.to_datetime(d.index);dr=pd.to_numeric(d.close,errors='coerce').pct_change();dv=dr.rolling(20,min_periods=15).var().replace(0,np.nan);dxy=pd.DataFrame({a:ret[a].rolling(20,min_periods=15).cov(dr)/dv.reindex(ret.index) for a in A})
v=get_index_daily_data('VIX',5000).set_index('date');v.index=pd.to_datetime(v.index);vr=pd.to_numeric(v.close,errors='coerce').pct_change();up=vr.where(vr>0);dn=vr.where(vr<0);vix=pd.DataFrame({a:ret[a].rolling(60,min_periods=25).cov(up)/up.rolling(60,min_periods=25).var()-ret[a].rolling(60,min_periods=25).cov(dn)/dn.rolling(60,min_periods=25).var() for a in A})
kurt=-ret.rolling(40,min_periods=30).kurt()
libs={'ravmom':trend,'risk_adjusted_trend':trend,'volnorm_reversal':rev,'realized_volatility':vol,'relative_volume':rv,'orthogonal_acceleration':orth,'negative_spx_beta':-beta,'dxy_beta':dxy,'vix_asymmetric_shock_beta':vix,'inverse_excess_kurtosis':kurt}
def stats(h):
 fw=panel.shift(-h)/panel-1; rows=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(rows));sd=x.std(ddof=1);turn=[]
 for i in range(1,len(f)):
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 regs={}
 for n,m in {'2020_2022':x.index.year<=2022,'2023_2024':x.index.year.isin([2023,2024]),'2025_2026':x.index.year.isin([2025,2026]),'2027':x.index.year==2027}.items():
  q=x[m];regs[n]={'dates':len(q),'ic':None if len(q)==0 else float(q.mean()),'icir':None if len(q)<2 or q.std(ddof=1)==0 else float(q.mean()/q.std(ddof=1)),'hit_ratio':None if len(q)==0 else float((q>0).mean())}
 return {'horizon':h,'dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'mean_rank_turnover':float(np.mean(turn)),'regimes':regs}
print('DATA',panel.index.min().date(),panel.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'of',f.size,'coverage',float(f.notna().mean().mean()))
for h in [1,5,10,20]:print('METRIC',json.dumps(stats(h),sort_keys=True))
cor={};mx=-1;who=None
for n,x in libs.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman') if len(z)>2 else np.nan;cor[n]={'rho':None if pd.isna(rho) else float(rho),'cells':len(z)}
 if not pd.isna(rho) and abs(rho)>mx:mx=abs(rho);who=n
print('LIBRARY',json.dumps(cor,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)
