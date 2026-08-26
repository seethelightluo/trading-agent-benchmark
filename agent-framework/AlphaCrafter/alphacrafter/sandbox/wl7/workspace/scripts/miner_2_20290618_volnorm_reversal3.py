import glob, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2029-06-18')
prices={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); d.date=pd.to_datetime(d.date); d=d[d.date<=CUT]; prices[s]=d.sort_values('date').set_index('date').close
c=pd.DataFrame(prices).sort_index(); r=c.pct_change(); vol=r.rolling(20).std();
f=(-r.rolling(3).sum()/(vol*np.sqrt(3)+1e-12)).shift(1); fw=c.shift(-10)/c-1
out=[]; sig=[]
for dt in c.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; out.append((dt,ic,len(z))); sig += [(dt,s,float(f.loc[dt,s])) for s in z.index]
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); m=r.ic.mean(); ir=m/r.ic.std(ddof=1)*np.sqrt(252)
print('factor=volnorm_reversal3 dates=%d range=%s..%s avg_n=%.2f coverage=%.2f IC=%.6f daily_ICIR=%.6f hit=%.4f'%(len(r),r.index.min().date(),r.index.max().date(),r.n.mean(),r.n.mean()/15,m,ir,(r.ic>0).mean()))
for nm,mask in [('2020-2024',r.index<'2025-01-01'),('2025-2026',(r.index>='2025-01-01')&(r.index<'2027-01-01')),('2027-2028',(r.index>='2027-01-01')&(r.index<'2029-01-01')),('since2028-09',r.index>='2028-09-01')]:
 q=r.loc[mask]; print(nm,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20290618_volnorm_reversal3_signal.csv',index=False)
