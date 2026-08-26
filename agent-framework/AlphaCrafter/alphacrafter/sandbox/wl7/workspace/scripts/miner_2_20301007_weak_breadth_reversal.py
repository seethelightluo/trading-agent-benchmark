import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r5=p.pct_change(5); r20=p.pct_change(20); vol=r.rolling(20).std(); breadth=(r20>0).mean(axis=1)
f=((-r5/vol.replace(0,np.nan)).mul(1.5-0.5*breadth,axis=0)).shift(1); fr=p.shift(-10)/p-1
rows=[]; sig=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); sig.append((dt,*[f.loc[dt,s] for s in U]))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); v=ic.ic.dropna(); n=len(v)
print('dates',n,'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15); print('IC10',v.mean(),'ICIR10',v.mean()/v.std(ddof=1)*np.sqrt(252/10),'hit',(v>0).mean())
for h in [1,5,20,40]:
 ff=p.shift(-h)/p-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
for name,sub in [('early',v.iloc[:n//3]),('middle',v.iloc[n//3:2*n//3]),('late',v.iloc[2*n//3:])]: print(name,sub.mean(),len(sub))
rank=f.rank(axis=1,pct=True); print('rank_turnover',((rank-rank.shift(1)).abs().mean(axis=1)).mean())
pd.DataFrame(sig,columns=['date']+U).set_index('date').to_csv('scripts/miner_2_20301007_weak_breadth_reversal_signal.csv'); ic.to_csv('scripts/miner_2_20301007_weak_breadth_reversal_ic.csv')
