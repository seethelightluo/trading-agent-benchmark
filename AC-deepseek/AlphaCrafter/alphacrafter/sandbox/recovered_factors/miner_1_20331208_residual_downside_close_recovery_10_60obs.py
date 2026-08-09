"""Miner 1: common-trend residual downside intraday recovery improvement, 10/60 observations."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; raw={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
def panel(col): return pd.concat({a:raw[a][col].astype(float) for a in A},axis=1).sort_index()
c,h,l=panel('close'),panel('high'),panel('low'); r=c.pct_change(); common=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(common,axis=0)); loc=(c-l).div((h-l).replace(0,np.nan)).clip(0,1)
# Relative quality of closes on asset-specific idiosyncratic down days: recent recovery vs its own baseline
q=loc.where(res.shift(1)<0)
f=(q.rolling(10,min_periods=4).mean()-q.rolling(60,min_periods=12).mean()).replace([np.inf,-np.inf],np.nan)
def test(hz):
 fw=c.shift(-hz).div(c)-1; out=[]; ns=[]
 for dt in f.index:
  z=pd.concat((f.loc[dt].rename('f'),fw.loc[dt].rename('y')),axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: out.append((dt,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); return x,dict(ic=x.mean(),icir=x.mean()/x.std(ddof=1),hit=(x>0).mean(),dates=len(x),mean_names=np.mean(ns),min_names=min(ns))
print('FACTOR residual_downside_close_recovery_improvement_10_60obs')
print('VISIBLE',c.index.min().date(),c.index.max().date(),'assets',len(A))
allx={}
for hz in (1,5,10,20):
 x,m=test(hz);allx[hz]=x;print('HORIZON',hz,{k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in m.items()},'last_complete',x.index.max().date())
x=allx[10]
for name,mask in [('2026_2029',(x.index>='2026-01-01')&(x.index<'2030-01-01')),('2030_2032',(x.index>='2030-01-01')&(x.index<'2033-01-01')),('2033_current',x.index>='2033-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
 y=x[mask];print('REGIME',name,{'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std(ddof=1)),'hit':float((y>0).mean())})
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('DIAGNOSTICS',{'coverage':float(f.notna().mean().mean()),'cells':int(f.notna().sum().sum()),'turnover':float(np.mean(ts)),'median_iqr':float(iqr.median()),'constant_dates':int((iqr<=1e-12).sum())})
