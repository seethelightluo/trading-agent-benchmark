import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index() for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:'2031-04-02'].ffill(); r=C.pct_change()
# Residual short-term reversal: subtract cross-sectional/common 5d move from each asset 5d return, negate residual.
ret=C.pct_change(5); common=ret.median(axis=1); raw=ret.sub(common,axis=0)*-1; sig=raw.rank(axis=1,pct=True).shift(1)
for h in [5,10,20]:
 rows=[]
 for i in range(len(C)-h):
  z=pd.concat([sig.iloc[i],C.iloc[i+h]/C.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(rows,columns=['date','n','ic'])
 def met(q): return q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q))
 for lab,q in [('full',a),('recent365',a[a.date>=a.date.max()-pd.Timedelta(days=365)]),('recent730',a[a.date>=a.date.max()-pd.Timedelta(days=730)]),('recent1095',a[a.date>=a.date.max()-pd.Timedelta(days=1095)])]: print(f'h{h} {lab}: dates={len(q)} avgN={q.n.mean():.2f} IC={met(q)[0]:.6f} ICIR={met(q)[1]:.6f} hit={(q.ic>0).mean():.4f}')
print(f'coverage={sig.notna().mean().mean():.4f} turnover={sig.diff().abs().mean(axis=1).mean():.6f} instruments={len(U)} cutoff={C.index[-1].date()}')
