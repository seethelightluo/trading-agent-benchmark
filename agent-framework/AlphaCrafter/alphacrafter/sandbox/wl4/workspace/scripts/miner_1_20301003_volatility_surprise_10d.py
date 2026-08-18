import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-10-02'); base='../persistent/stock_data'
d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:end].ffill(); r=C.pct_change()
# Volatility surprise: recent 5-session realized volatility relative to 60-session baseline,
# shifted one completed session; positive score means unusually high current volatility.
sig=(r.rolling(5,min_periods=4).std()/r.rolling(60,min_periods=40).std()).shift(1)
for h in [5,10,20]:
 rows=[]
 for i in range(len(C)-h):
  z=pd.concat([sig.iloc[i],C.iloc[i+h]/C.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(rows,columns=['date','n','ic']); rec=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
 def met(q): return (q.ic.mean(), q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)))
 print(f'h{h}: dates={len(a)} avgN={a.n.mean():.2f} IC={met(a)[0]:.6f} ICIR={met(a)[1]:.6f} hit={(a.ic>0).mean():.4f} recentIC={met(rec)[0]:.6f} recentICIR={met(rec)[1]:.6f}')
print(f'coverage={sig.notna().mean().mean():.4f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(U)} cutoff={C.index[-1].date()}')
# regime split by cross-sectional median 20d return
mkt=r.mean(axis=1).rolling(20).mean().shift(1)
for label,mask in [('risk_on',mkt>0),('risk_off',mkt<=0)]:
 rows=[]
 for i in range(len(C)-10):
  if not bool(mask.iloc[i]): continue
  z=pd.concat([sig.iloc[i],C.iloc[i+10]/C.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(f'{label}: dates={len(rows)} IC={np.mean(rows) if rows else np.nan:.6f}')
