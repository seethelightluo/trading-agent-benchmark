import numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT='../persistent/stock_data'; A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index() for a in A}; ix=sorted(set().union(*[set(x.index) for x in D.values()]))
def pan(k):return pd.DataFrame({a:D[a].reindex(ix)[k] for a in A})
c,h,l,v=pan('close'),pan('high'),pan('low'),pan('volume');r=c.pct_change(fill_method=None);vis=c.index[c.index<='2028-09-20'];med=r.median(axis=1)
def corr(x,y,w,mask=None):
 if mask is not None:x=x.where(mask,axis=0);y=y.where(mask)
 return x.rolling(w,min_periods=12).corr(y)
# Candidate: downside range pressure, cross-sectionally residualized each date on its known technical overlaps.
rng=(h-l).div(c.shift(1)).abs(); raw=(rng*(-r).clip(lower=0)).rolling(20,min_periods=15).mean().div(rng.rolling(20,min_periods=15).mean())
L={
'volnorm_reversal':-r.rolling(5,min_periods=4).sum()/r.rolling(20,min_periods=15).std(),
'return_sign_balance':(r>0).rolling(20,min_periods=15).mean(),
'dispersion_sensitivity':corr(r,r.std(axis=1),20), 'vol_cluster':corr(r.abs(),r.abs().shift(),20),
'corr_asym':corr(r,med,60,med<0)-corr(r,med,60,med>=0),
'risk_adjusted_trend':(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std(),
'relative_volume':np.log(v/v.rolling(20,min_periods=15).mean()), 'persist':corr(r,r.shift(),20),
'efficiency':r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum(),
'liquidity_stress':np.log((r.abs()/v).rolling(20,min_periods=15).mean()/(r.abs()/v).rolling(60,min_periods=15).mean()),
'downvolume':np.log(v.where(r<0).rolling(60,min_periods=12).mean()/v.where(r>=0).rolling(60,min_periods=12).mean())}
L['trend_accel']=L['risk_adjusted_trend']-(c/c.shift(60)-1)/r.rolling(60,min_periods=15).std();L['downvolume_accel']=L['downvolume']-np.log(v.where(r<0).rolling(20,min_periods=12).mean()/v.where(r>=0).rolling(20,min_periods=12).mean())
basis=['efficiency','volnorm_reversal','vol_cluster','risk_adjusted_trend','return_sign_balance','persist']
f=pd.DataFrame(np.nan,index=ix,columns=A)
for t in ix:
 z=pd.concat([raw.loc[t]]+[L[k].loc[t] for k in basis],axis=1).dropna()
 if len(z)>=8:
  # standardized OLS residual eliminates contemporaneous linear overlap while retaining cross-sectional idiosyncratic pressure.
  y=(z.iloc[:,0]-z.iloc[:,0].mean())/z.iloc[:,0].std(); X=z.iloc[:,1:].apply(lambda x:(x-x.mean())/x.std())
  f.loc[t,z.index]=y-np.c_[np.ones(len(X)),X].dot(np.linalg.lstsq(np.c_[np.ones(len(X)),X],y,rcond=None)[0])
def report(sub,H):
 q=c.shift(-H)/c-1;z=[];nn=[]
 for t in sub:
  x=pd.concat([f.loc[t],q.loc[t]],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);nn.append(len(x))
 z=np.array(z);return len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),np.mean(nn)
for H in [1,5,10,20]:
 n,ic,ir,hit,ni=report(vis,H);print('H',H,'dates',n,'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4),'instruments',round(ni,2),'coverage',round(n*ni/(len(vis)*15),4))
for name,x in L.items():
 z=[]
 for t in vis:
  q=pd.concat([f.loc[t],x.loc[t]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('CORR',name,'dates',len(z),'max',round(max(abs(np.array(z))),6) if z else None,'mean',round(np.mean(z),6) if z else None)
for name,sub in [('early',vis[vis<'2024-01-01']),('mid',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('late',vis[vis>='2026-01-01'])]:
 n,ic,ir,hit,ni=report(sub,5);print('REGIME',name,'dates',n,'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4),'instruments',round(ni,2))
print('period',vis[0],vis[-1],'cells',int(f.loc[vis].notna().sum().sum()))
