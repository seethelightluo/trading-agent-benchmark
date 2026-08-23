import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-10-30')
cl={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
# Dispersion-conditioned short reversal: reverse 5d return when lagged cross-asset dispersion is elevated.
csdisp=r.rolling(20,min_periods=15).apply(lambda z: np.nanstd(z),raw=True)
# rowwise dispersion across assets, using lagged daily cross-sectional std then 20d smoothing
csdisp= r.std(axis=1).rolling(20,min_periods=15).mean()
threshold=csdisp.rolling(252,min_periods=126).quantile(.65)
active=(csdisp>threshold).astype(float)
f=(-r.rolling(5,min_periods=5).sum()).mul(active,axis=0)
D=[]; A=[]; N=[]; C=[]; prev=None; T=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]); A.append(z); N.append(int(ok.sum())); C.append(float(ok.mean()))
  if prev is not None:
   oo=x[oo] if False else x.notna()&prev.notna(); T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); A=np.array(A)
def stat(mask):
 z=A[mask]; return (len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)) if len(z)>1 else np.nan,float((z>0).mean()) if len(z) else np.nan)
print({'factor':'dispersion_conditioned_5d_reversal','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(N)),'coverage':float(np.mean(C)),'ic':float(A.mean()),'icir':float(A.mean()/A.std(ddof=1)),'hit':float((A>0).mean()),'turnover':float(np.mean(T))})
for name,m in [('recent180',D>=pd.Timestamp('2031-01-01')),('recent360',D>=pd.Timestamp('2030-10-01')),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('recent60',D>=pd.Timestamp('2031-07-01'))]: print(name,stat(m))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in D:
  x=f.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20311030_dispersion_reversal_signal.csv')
pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_3_20311030_dispersion_reversal_ic.csv',index=False)
