import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-11-13')
Dct={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}; p=pd.DataFrame({s:Dct[s].close for s in U}).sort_index(); v=pd.DataFrame({s:Dct[s].volume for s in U}).reindex(p.index); r=p.pct_change()
# Volume-confirmed 20d momentum: medium-term return weighted by log volume acceleration, all lag-safe.
vr=np.log1p(v).rolling(20,min_periods=15).mean()/np.log1p(v).rolling(60,min_periods=40).mean(); f=r.rolling(20,min_periods=20).sum()*vr
D=[]; A=[]; N=[]; C=[]; T=[]; prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2021-01-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]); A.append(z); N.append(ok.sum()); C.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna(); T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); A=np.array(A)
def st(m):
 z=A[m]; return len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean())
print({'factor':'volume_confirmed_20d_momentum','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(N)),'coverage':float(np.mean(C)),'ic':float(A.mean()),'icir':float(A.mean()/A.std(ddof=1)),'hit':float((A>0).mean()),'turnover':float(np.mean(T))})
for n,m in [('recent180',D>=D[-1]-pd.Timedelta(days=270)),('recent360',D>=D[-1]-pd.Timedelta(days=540)),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',D>=pd.Timestamp('2031-01-01'))]: print(n,st(m))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in D:
  x=f.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20311113_volume_confirmed_signal.csv'); pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_3_20311113_volume_confirmed_ic.csv',index=False)
