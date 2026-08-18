import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
CURRENT=pd.Timestamp('2034-02-02')
files=glob.glob('../persistent/stock_data/*.csv'); px={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); dc={c.lower():c for c in d.columns}
 if 'date' in dc and 'close' in dc:
  q=pd.Series(d[dc['close']].values,index=pd.to_datetime(d[dc['date']])).sort_index(); px[s]=q[q.index<=CURRENT]
P=pd.DataFrame(px).sort_index().ffill(); rets=P.pct_change(); r60=P/P.shift(60)-1
cons=(rets.gt(0).rolling(60).mean()-0.5)*2; vol=rets.rolling(30).std()*np.sqrt(252)
F=(r60*cons/vol).shift(1); fr=P.shift(-10)/P-1
ics=[]; used=[]; turnovers=[]; prev=None
for dt in F.index:
 x=F.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic); used.append((dt,len(z)))
  ranks=x.rank(pct=True)
  if prev is not None: turnovers.append((ranks-prev).abs().mean())
  prev=ranks
arr=np.array(ics); mean=arr.mean(); sd=arr.std(ddof=1); print('factor=60d return*directional consistency/30d vol, lag1 H10'); print('dates',len(arr),'avg_n',np.mean([n for _,n in used]),'start',used[0][0].date(),'end',used[-1][0].date()); print('IC',mean,'ICIR_daily',mean/sd,'annualized_ICIR',mean/sd*np.sqrt(252),'hit',np.mean(arr>0),'turnover',np.nanmean(turnovers),'coverage',np.mean([n/len(P.columns) for _,n in used]))
for n in [120,260,520,780]:
 a=arr[-n:]; print('recent',len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),np.mean(a>0))
pd.DataFrame(F).to_csv('scripts/artifacts/miner_1_20340202_consistent_trend_signal.csv'); pd.DataFrame({'date':[d for d,n in used],'ic':ics,'n':[n for d,n in used]}).to_csv('scripts/artifacts/miner_1_20340202_consistent_trend_ic.csv',index=False)
