import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
p=pd.DataFrame({os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float) for f in glob.glob('../persistent/stock_data/*.csv')}).sort_index().loc[:'2034-07-19']
r=p.pct_change(); lag=r.shift(1); mom=p.shift(1).pct_change(40); down=np.minimum(lag,0).rolling(40).std()*np.sqrt(40); f=(mom/(down+0.001)).rolling(3).mean()
rows=[]
for dt in p.index:
 if dt not in f.index:continue
 x=f.loc[dt]; y=p.shift(-10).loc[dt]/p.loc[dt]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((dt,spearmanr(x[ok],y[ok]).statistic,int(ok.sum())))
a=np.array([q[1] for q in rows]);print('dates',len(a),'avgN',np.mean([q[2] for q in rows]),'coverage',f.notna().stack().mean());print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2025','2029-12-31'),('2030','2034-07-19'),('2033','2034-07-19')]:
 q=np.array([z[1] for z in rows if pd.Timestamp(lo)<=z[0]<=pd.Timestamp(hi)]);print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 z=[]
 for dt in p.index:
  if dt not in f.index:continue
  x=f.loc[dt];y=p.shift(-h).loc[dt]/p.loc[dt]-1;ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.mean(z))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20340720_downside_adjusted_trend_ic.csv',index=False)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340720_downside_adjusted_trend_signal.csv',index=False)
