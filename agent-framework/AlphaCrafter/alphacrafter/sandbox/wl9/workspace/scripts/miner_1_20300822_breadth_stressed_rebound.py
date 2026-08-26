import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-08-22')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
ret10=p.pct_change(10); med10=ret10.median(axis=1)
down=r.clip(upper=0).abs().rolling(30,min_periods=15).mean()
# Contrarian rebound signal: relative 10d shock, only during broad weak tape.
shock=-(ret10.sub(med10,axis=0)).div(down+1e-8)
breadth=(p.pct_change(20)>0).mean(axis=1)
gate=(breadth<0.47).astype(float)
f=shock.mul(gate,axis=0).rolling(3,min_periods=1).mean().shift(1)
y=p.shift(-60)/p-1
rows=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((dt,q,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n'])
ic=a.ic.mean(); ir=ic/(a.ic.std(ddof=1)+1e-12)
print('horizon 60 dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
for name,sl in [('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
 if len(sl): print('regime',name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
rank=f.rank(axis=1,pct=True); print('turnover_proxy',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_1_20300822_breadth_stressed_rebound_signal.csv')
