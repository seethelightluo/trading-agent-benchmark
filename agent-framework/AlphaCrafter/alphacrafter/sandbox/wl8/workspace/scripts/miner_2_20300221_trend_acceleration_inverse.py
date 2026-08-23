import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2030-02-20'); frames={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); frames[s]=d[d.date<=END].set_index('date')
rows=[]
for s,d in frames.items():
 r=d.close.pct_change(); base=d.close.pct_change(60)-d.close.pct_change(10); vol=r.rolling(40).std()
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'base':base,'vol':vol,'fwd20':d.close.shift(-20)/d.close-1,'fwd10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True).sort_values(['symbol','date']); x['med']=x.groupby('date').base.transform('median')
x['signal']=(-(x.base-x.med)/x.vol).groupby(x.symbol).shift(1); x=x.sort_values('date')
for h in [10,20]:
 out=[]
 for dt,g in x.groupby('date'):
  z=g[['signal',f'fwd{h}']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.signal,z[f'fwd{h}']).statistic,len(z)))
 ic=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); print('H',h,'dates',len(ic),'avg_n',ic.n.mean(),'IC',m,'ICIR',m/ic.ic.std(ddof=1),'hit',(ic.ic>0).mean())
 for a,b in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2028-12-31'),('2029-01-01','2030-02-20')]:
  q=ic.loc[a:b]
  if len(q)>2: print('REG',a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
 if h==20: x[['date','symbol','signal']].dropna().to_csv('scripts/miner_2_20300221_trend_acceleration_inverse_signal.csv',index=False)
print('coverage',x.signal.notna().mean())
w=x.pivot(index='date',columns='symbol',values='signal'); r=w.rank(axis=1,pct=True); print('turnover',((r-r.shift()).abs().mean(axis=1)).mean())
