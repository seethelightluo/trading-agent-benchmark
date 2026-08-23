import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-11')
cl={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); base=disp.rolling(252,min_periods=126).median()
# Continuous, lagged regime weight: trend is emphasized in orderly markets but never blanked.
w=(base/disp).clip(0.35,2.5)
f=r.rolling(20,min_periods=20).sum().shift(1).mul(w.shift(1),axis=0)
D=[]; A=[]; N=[]; C=[]; T=[]; prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8 or x[ok].nunique()<3 or y[ok].nunique()<3: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]); A.append(z); N.append(int(ok.sum())); C.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna(); T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); A=np.array(A)
def stat(mask):
 z=A[mask]; return (len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)) if len(z)>1 else np.nan,float((z>0).mean()) if len(z) else np.nan)
print({'factor':'continuous_dispersion_weighted_trend_20d','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(N)),'coverage':float(np.mean(C)),'ic':float(A.mean()),'icir':float(A.mean()/A.std(ddof=1)),'hit':float((A>0).mean()),'turnover':float(np.mean(T))})
for name,m in [('recent180',D>=pd.Timestamp('2031-01-01')),('recent90',D>=pd.Timestamp('2031-06-01')),('recent60',D>=pd.Timestamp('2031-08-01')),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',D>=pd.Timestamp('2031-01-01'))]: print(name,stat(m))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in D:
  x=f.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>=3:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20311211_continuous_dispersion_weighted_trend_signal.csv')
pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_3_20311211_continuous_dispersion_weighted_trend_ic.csv',index=False)
