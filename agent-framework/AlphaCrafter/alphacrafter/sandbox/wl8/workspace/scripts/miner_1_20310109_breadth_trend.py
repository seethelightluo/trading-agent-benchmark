import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); mom=p/p.shift(40)-1; bread=(p.pct_change(20)>0).mean(axis=1); f=mom*(0.5+(bread-0.5).clip(-0.5,0.5))
D=[];A=[];N=[];C=[];T=[];prev=None
for i in range(40,len(p)-10):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or d>pd.Timestamp('2030-12-11'): continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 ic=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(ic):
  D.append(d);A.append(ic);N.append(int(ok.sum()));C.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna();T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
a=np.array(A);D=pd.DatetimeIndex(D)
print({'factor':'breadth_gated_trend_40d_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(np.mean(N),2),'coverage':round(np.mean(C),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'turnover':round(np.mean(T),6)})
for l,m in [('180',D>=pd.Timestamp('2030-07-01')),('90',D>=pd.Timestamp('2030-10-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(l,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20310109_breadth_trend_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_1_20310109_breadth_trend_signal.csv')
