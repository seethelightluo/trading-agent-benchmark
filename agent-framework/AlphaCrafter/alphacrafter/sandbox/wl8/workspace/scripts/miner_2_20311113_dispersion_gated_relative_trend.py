import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-10-30')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change(); disp=r.std(axis=1)
mom=r.rolling(15,min_periods=10).sum(); rel=mom.sub(mom.mean(axis=1),axis='index'); gate=(disp>disp.rolling(60,min_periods=30).median()); f=rel.where(gate, np.nan).shift(1)
D=[];ics=[];ns=[];cvs=[];turns=[];prev=None
for i in range(len(p)-10):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(d);ics.append(z);ns.append(int(ok.sum()));cvs.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna(); turns.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); a=np.array(ics)
print({'factor':'dispersion_gated_relative_trend_15d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cvs)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for name,mask in [('365',D>=pd.Timestamp('2030-10-30')),('180',D>=pd.Timestamp('2031-04-30')),('60',D>=pd.Timestamp('2031-08-30')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',(D>=pd.Timestamp('2030-01-01'))&(D<pd.Timestamp('2031-01-01'))),('2031YTD',D>=pd.Timestamp('2031-01-01'))]:
 z=a[mask]; print(name,len(z),float(z.mean()) if len(z) else np.nan,float(z.mean()/z.std(ddof=1)) if len(z)>1 else np.nan)
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in D:
  x=f.loc[d]; y=yy.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,float(np.nanmean(q)),len(q))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20311113_dispersion_gated_relative_trend_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_2_20311113_dispersion_gated_relative_trend_ic.csv',index=False)
