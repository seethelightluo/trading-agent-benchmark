import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=120:
  x=d.set_index('date'); px[s]=x.close.astype(float); vol[s]=x.volume.astype(float)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
R=P.pct_change(); market=R.mean(axis=1)
# Residual reversal: recent 20d asset return net of contemporaneous cross-asset market,
# scaled by idiosyncratic volatility; buying residual losers exploits relative mean reversion.
beta=R.rolling(60,min_periods=40).cov(market).div(market.rolling(60,min_periods=40).var(),axis=0)
resid=R.sub(beta.mul(market,axis=0),axis=0)
res20=resid.rolling(20,min_periods=15).sum()
rv=resid.rolling(40,min_periods=25).std()*np.sqrt(40)
f=(-res20/(rv+1e-8)).clip(-8,8)
fw=P.shift(-10)/P-1
a=[]; ds=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): a.append(c); ds.append(dt); ns.append(len(z))
a=np.array(a); ds=pd.DatetimeIndex(ds)
print('dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'daily_ICIR',round(a.mean()/a.std(ddof=1),6),'annualized_ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
for h in [5,10,20]:
 fw2=P.shift(-h)/P-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw2.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): aa.append(c)
 print('decay',h,round(np.mean(aa),6),len(aa))
for x,y in [('2020-01-01','2023-12-31'),('2024-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-04-01')]:
 z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
 if len(z)>1: print('regime',x,y,'dates',len(z),'IC',round(z.mean(),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in ds],index=ds)
print('turnover',round(S.diff().abs().mean().mean(),6))
rows=[(dt,s,float(f.loc[dt,s])) for dt in f.index for s in f.columns if pd.notna(f.loc[dt,s])]
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340413_residual_reversal_signal.csv',index=False)
