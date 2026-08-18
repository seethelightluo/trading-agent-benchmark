import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; end=pd.Timestamp('2031-02-19')
d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:end].ffill(); r=C.pct_change()
# Relative-strength quality: 20d return in excess of contemporaneous cross-asset median,
# multiplied by persistence (share of positive daily returns), then volatility scaled.
ret20=C.pct_change(20); med=ret20.median(axis=1); excess=ret20.sub(med,axis=0)
persist=r.rolling(20,min_periods=15).apply(lambda x: np.mean(x>0),raw=True)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
raw=(excess/vol)*(0.5+persist)
sig=raw.rank(axis=1,pct=True).shift(1)
rows=[]
for i in range(len(C)-10):
 z=pd.concat([sig.iloc[i],C.iloc[i+10]/C.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']);
def met(q): return (q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)),(q.ic>0).mean()) if len(q)>1 else (np.nan,np.nan,np.nan)
for label,q in [('full',a),('recent365',a[a.date>=a.date.max()-pd.Timedelta(days=365)]),('recent730',a[a.date>=a.date.max()-pd.Timedelta(days=730)])]: print(label,'dates',len(q),'avgN',round(q.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%met(q))
print('coverage %.4f turnover %.6f instruments %d cutoff %s'%(sig.notna().mean().mean(),sig.diff().abs().mean(axis=1).mean(),len(U),C.index[-1].date()))
# regime split by median cross-sectional 20d return sign
for name,q in [('risk_on',a[med.reindex(a.date).values>0]),('risk_off',a[med.reindex(a.date).values<=0])]: print(name,'dates',len(q),'IC %.6f ICIR %.6f'%met(q)[:2])
