"""Miner 1 candidate: residual downside semivolatility compression 20/60 days.
High values mean an asset's recent idiosyncratic downside amplitude has fallen
relative to its own 60-day baseline.  All inputs use completed daily bars only.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; raw={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
c=pd.concat({a:raw[a].close.astype(float) for a in A},axis=1).sort_index()
r=c.pct_change(); common=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(common,axis=0))
# RMS negative residual return; signal is its proportional compression versus slow baseline.
down=(-res.clip(upper=0)).pow(2)
short=down.rolling(20,min_periods=12).mean().pow(.5)
long=down.rolling(60,min_periods=35).mean().pow(.5)
f=(1-short.div(long)).replace([np.inf,-np.inf],np.nan)
def ev(h):
 y=c.shift(-h).div(c)-1; obs=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   obs.append((d,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(obs)); return x,{'ic':x.mean(),'icir':x.mean()/x.std(ddof=1),'hit':(x>0).mean(),'dates':len(x),'mean_names':np.mean(ns),'min_names':min(ns)}
print('FACTOR residual_downside_semivol_compression_20_60d')
print('EXPRESSION 1 - rms_20(min(residual_return,0))/rms_60(min(residual_return,0))')
print('VISIBLE',c.index.min().date(),c.index.max().date(),'assets',len(A))
xs={}
for h in (1,5,10,20):
 xs[h],m=ev(h);print('HORIZON',h,m)
x=xs[10]
for n,mask in [('2026_2029',(x.index>='2026')&(x.index<'2030')),('2030_2032',(x.index>='2030')&(x.index<'2033')),('2033_current',x.index>='2033'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
 z=x[mask];print('REGIME',n,{'dates':len(z),'ic':z.mean(),'icir':z.mean()/z.std(ddof=1),'hit':(z>0).mean()})
rk=f.rank(axis=1,pct=True); tr=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:tr.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('DIAGNOSTICS',{'coverage':f.notna().mean().mean(),'valid_cells':int(f.notna().sum().sum()),'turnover':np.mean(tr),'median_iqr':iqr.median(),'constant_dates':int((iqr<=1e-12).sum())})
print('LIBRARY_CORRELATION deferred pending IC gates')
