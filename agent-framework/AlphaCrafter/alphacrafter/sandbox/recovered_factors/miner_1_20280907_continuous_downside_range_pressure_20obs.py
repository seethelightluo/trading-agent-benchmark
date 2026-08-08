import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT='../persistent/stock_data'; A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index() for a in A}; ix=sorted(set().union(*[set(x.index) for x in D.values()]))
def pan(k):return pd.DataFrame({a:D[a].reindex(ix)[k] for a in A})
c,h,l,v=pan('close'),pan('high'),pan('low'),pan('volume'); r=c.pct_change(fill_method=None); vis=c.index[c.index<='2028-09-06']
# One idea: continuous downside-range pressure: fraction of 20-observation range activity that occurs on adverse, large-return days.
rng=(h-l).div(c.shift(1)).abs(); down=(-r).clip(lower=0); f=(rng*down).rolling(20,min_periods=15).mean().div(rng.rolling(20,min_periods=15).mean())
# Independently reconstruct core / closest admitted factor signals to quantify redundancy.
med=r.median(axis=1)
def corr(x,y,w,mask=None):
 if mask is not None:x=x.where(mask,axis=0);y=y.where(mask)
 return x.rolling(w,min_periods=12).corr(y)
def beta(x,y,w=60):return x.rolling(w,min_periods=12).cov(y).div(y.rolling(w,min_periods=12).var(),axis=0)
L={
'volnorm_reversal':-r.rolling(5,min_periods=4).sum()/r.rolling(20,min_periods=15).std(),
'return_sign_balance':(r>0).rolling(20,min_periods=15).mean(),
'dispersion_sensitivity':corr(r,r.std(axis=1),20),
'vol_cluster':corr(r.abs(),r.abs().shift(),20),
'corr_asym':corr(r,med,60,med<0)-corr(r,med,60,med>=0),
'risk_adjusted_trend':(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std(),
'relative_volume':np.log(v/v.rolling(20,min_periods=15).mean()),
'persist':corr(r,r.shift(),20),
'efficiency':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),
'liquidity_stress':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=15).mean()),
'downvolume':np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean()),
}
L['trend_accel']=L['risk_adjusted_trend']-(c/c.shift(60)-1)/r.rolling(60,min_periods=15).std();L['downvolume_accel']=L['downvolume']-np.log(v.where(r<0).rolling(20,min_periods=12).mean()/v.where(r>=0).rolling(20,min_periods=12).mean())
for H in [1,5,10,20]:
 q=c.shift(-H)/c-1; z=[];nn=[]
 for t in vis:
  x=pd.concat([f.loc[t],q.loc[t]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);nn.append(len(x))
 z=np.array(z);print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'instruments',round(np.mean(nn),2),'coverage',round(len(z)*np.mean(nn)/(len(vis)*15),4))
for n,x in L.items():
 z=[]
 for t in vis:
  q=pd.concat([f.loc[t],x.loc[t]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('CORR',n,'dates',len(z),'max',round(np.max(np.abs(z)),6) if z else None,'mean',round(np.mean(z),6) if z else None)
for label,subset in [('early',vis[vis<'2024-01-01']),('mid',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('late',vis[vis>='2026-01-01'])]:
 z=[];q=c.shift(-5)/c-1
 for t in subset:
  x=pd.concat([f.loc[t],q.loc[t]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 z=np.array(z);print('REGIME',label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('period',vis[0],vis[-1],'cells',int(f.loc[vis].notna().sum().sum()))
