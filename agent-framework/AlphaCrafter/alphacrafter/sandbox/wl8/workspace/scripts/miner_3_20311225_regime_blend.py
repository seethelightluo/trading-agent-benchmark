import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-25')
cl={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); base=disp.rolling(252,min_periods=126).median()
# Smooth bounded regime blend: low dispersion favors medium trend; high dispersion favors short reversal.
ratio=(disp/base).clip(.4,2.5); g=((ratio-0.8)/0.8).clip(0,1)
trend=r.rolling(20,min_periods=20).sum().shift(1)
rev=-r.rolling(5,min_periods=5).sum().shift(1)
f=trend.mul(1-g.shift(1),axis=0)+rev.mul(g.shift(1),axis=0)
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
def stat(m):
 z=A[m]; return (int(len(z)),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()))
print({'factor':'dispersion_regime_blended_trend_reversal_20_5d','dates':len(A),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(float(np.mean(N)),2),'coverage':round(float(np.mean(C)),4),'ic':round(float(A.mean()),6),'icir':round(float(A.mean()/A.std(ddof=1)),6),'hit':round(float((A>0).mean()),4),'turnover':round(float(np.mean(T)),6)})
for name,m in [('recent365',D>=pd.Timestamp('2030-12-25')),('recent180',D>=pd.Timestamp('2031-06-01')),('recent90',D>=pd.Timestamp('2031-09-01')),('2028',(D>=pd.Timestamp('2028-01-01'))&(D<pd.Timestamp('2029-01-01'))),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031',D>=pd.Timestamp('2031-01-01'))]: print(name,stat(m))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in D:
  x=f.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>=3:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,round(float(np.nanmean(q)),6),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_3_20311225_regime_blend_signal.csv')
pd.DataFrame({'date':D,'ic':A}).to_csv('scripts/miner_3_20311225_regime_blend_ic.csv',index=False)
