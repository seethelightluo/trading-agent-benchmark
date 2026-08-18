import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-01-22'); base='../persistent/stock_data'
d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:end].ffill(); r=C.pct_change()
# High-dispersion short-term reversal: buy recent losers only when cross-asset daily dispersion is elevated.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
th=disp.rolling(252,min_periods=126).median()
gate=(disp>th).astype(float)
raw=-C.pct_change(5).div(r.rolling(20,min_periods=15).std().replace(0,np.nan),axis=0).mul(gate,axis=0)
sig=raw.rank(axis=1,pct=True).shift(1)
rows=[]
for i in range(len(C)-10):
 z=pd.concat([sig.iloc[i],C.iloc[i+10]/C.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic'])
def met(q): return q.ic.mean(), q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q))
for label,q in [('full',a),('recent365',a[a.date>=a.date.max()-pd.Timedelta(days=365)]),('recent730',a[a.date>=a.date.max()-pd.Timedelta(days=730)])]: print(label,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(met(q)[0],met(q)[1],(q.ic>0).mean()))
print('avgN %.2f coverage %.4f turnover %.6f cutoff %s universe %d'%(a.n.mean(),sig.notna().mean().mean(),sig.diff().abs().mean(axis=1).mean(),C.index[-1].date(),len(U)))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2031')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; print(lo+'-'+hi,'dates',len(q),'IC %.6f'%q.ic.mean())
