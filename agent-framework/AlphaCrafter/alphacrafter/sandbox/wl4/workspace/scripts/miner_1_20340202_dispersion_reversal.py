import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
CURRENT=pd.Timestamp('2034-02-02'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 d=pd.read_csv(f); c={x.lower():x for x in d.columns}; s=os.path.basename(f)[:-4]
 if 'date' in c and 'close' in c: px[s]=pd.Series(d[c['close']].values,index=pd.to_datetime(d[c['date']])).sort_index()
P=pd.DataFrame(px).sort_index(); P=P[P.index<=CURRENT].ffill(); R=P.pct_change();
# residual short reversal only when cross-asset dispersion is elevated; volatility normalize
csdisp=R.std(axis=1).rolling(20).mean(); cutoff=csdisp.rolling(252).median()
rev=-(P/P.shift(5)-1)/(R.rolling(20).std()*np.sqrt(252)); F=rev.where(csdisp.shift(1)>cutoff.shift(1),0).shift(1); fr=P.shift(-10)/P-1
ics=[]; ns=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v): ics.append(v);ns.append(len(z))
a=np.array(ics); print('factor=volscaled 5d reversal active high dispersion, lag1 H10'); print('dates',len(a),'avg_n',np.mean(ns),'start',F.index[F.index.get_loc(F.index[0])].date(),'end',F.index[-1].date()); print('IC',a.mean(),'dailyICIR',a.mean()/a.std(ddof=1),'annualICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',np.mean(a>0),'coverage',np.mean(np.array(ns)/len(P.columns)))
for n in [120,260,520,780]:
 q=a[-n:]; print('recent',len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),np.mean(q>0))
pd.DataFrame(F).to_csv('scripts/artifacts/miner_1_20340202_dispersion_reversal_signal.csv'); pd.DataFrame({'ic':a,'n':ns}).to_csv('scripts/artifacts/miner_1_20340202_dispersion_reversal_ic.csv',index=False)
