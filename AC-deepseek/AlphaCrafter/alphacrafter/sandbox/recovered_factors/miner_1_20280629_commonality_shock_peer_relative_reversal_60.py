"""One idea: commonality-shock gated peer-relative reversal."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); C[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); r=P.pct_change(); m=r.median(axis=1); cutoff=P.dropna(how='all').index.max(); H=[1,5,10,20]
# A rising high-commonality state is a stress-transition condition. Within it, fade recent peer-relative moves scaled by own volatility.
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
corr=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
common=corr.median(axis=1); event=(common>common.rolling(60,min_periods=40).quantile(.75))&(common>common.shift(5))
raw=-(P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)/r.rolling(20,min_periods=15).std())
# Preserve an event signal for 60 observations, smoothing isolated high-correlation shocks.
f=raw.where(event,axis=0).rolling(60,min_periods=1).mean().where(event.rolling(60,min_periods=1).sum()>0,axis=0); f=f.sub(f.median(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
fw={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=fw[h].reindex(x.index);z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v);ns.append(len(q))
 z=np.array(z)
 if not len(z): return {'dates':0}
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR commonality_shock_peer_relative_reversal_60 cutoff',cutoff.date(),'assets',len(A),'EVENTS',int(event.sum()),'MEAN_COMMONALITY',round(float(common.mean()),6))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),5))
for h in H: print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01',str(cutoff.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# Reconstruct every admitted factor signal for binding full-library independence evidence.
vol=r.rolling(20,min_periods=15).std(); trend=P.pct_change(20)/vol
def beta(x,y,down=False,up=False,w=40):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1); z=z.where(z.y<0 if down else z.y>0 if up else pd.Series(True,index=z.index)); return z.x.rolling(w,min_periods=12).cov(z.y)/z.y.rolling(w,min_periods=12).var()
vx=get_index_daily_data('VIX',5000).copy();vx.date=pd.to_datetime(vx.date);vx=pd.to_numeric(vx.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill();vr=vx.pct_change();sg=pd.Series(np.where(vx/vx.shift(20)-1>0,-1.,1.),index=P.index)
dxy=get_index_daily_data('DXY',5000).copy();dxy.date=pd.to_datetime(dxy.date);dr=pd.to_numeric(dxy.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill().pct_change();neg=r.where(r<0,0);dvc=-np.log((np.sqrt((neg**2).rolling(10,min_periods=7).mean())+1e-5)/(np.sqrt((neg**2).rolling(40,min_periods=25).mean())+1e-5));dvc=dvc.sub(dvc.median(axis=1),axis=0);peak=P.rolling(60,min_periods=45).max();dd=P/peak-1;rec=(dd-dd.shift(10))/(.01-dd.shift(10));rec=rec.sub(rec.median(axis=1),axis=0)
S={'ravmom':trend,'volnormrev5':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'vixtrend':trend.mul(sg,axis=0),'downbeta':pd.DataFrame({a:beta(r[a],m,down=True) for a in A}),'idio':-r.sub(m,axis=0).rolling(20,min_periods=15).std(),'stableliq':-pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std() for a in A}),'skew':r.rolling(60,min_periods=40).skew(),'gradual':trend*np.tanh((-np.log(vol/r.rolling(40,min_periods=15).std())).clip(-2,2)),'dxyasym':pd.DataFrame({a:r[a].where(dr>0).rolling(60,min_periods=25).mean()-r[a].where(dr<=0).rolling(60,min_periods=25).mean() for a in A}),'poststress':r.sub(m,axis=0).where((m<m.rolling(60,min_periods=40).quantile(.25)).shift(1),axis=0).rolling(60,min_periods=12).median(),'delayedstress':r.sub(m,axis=0).where((m<m.rolling(60,min_periods=40).quantile(.25)).shift(2),axis=0).rolling(60,min_periods=12).median()}
mx=0;who='';cells=0
for n,g in S.items():
 q=pd.concat([f.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic; print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6))
 if abs(rho)>mx: mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',cells,'N_FACTORS',len(S))
