import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; end=pd.Timestamp('2030-07-24')
px={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').close.astype(float) for s in U}; P=pd.DataFrame(px).sort_index().loc[:end].ffill(); r=P.pct_change()
# Reversal of recent move, emphasized when short volatility is elevated versus medium volatility.
vr=r.rolling(5,min_periods=5).std()/r.rolling(40,min_periods=30).std(); sig=(-P.pct_change(5)*vr).shift(1)
for h in [5,10,20]:
 rows=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(rows,columns=['date','n','ic']); rec=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
 def m(q): return q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q))
 print(f'h{h}: dates={len(a)} avgN={a.n.mean():.2f} IC={m(a)[0]:.6f} ICIR={m(a)[1]:.6f} hit={(a.ic>0).mean():.4f} recentIC={m(rec)[0]:.6f} recentICIR={m(rec)[1]:.6f}')
print(f'coverage={sig.notna().mean().mean():.4f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} cutoff={P.index[-1].date()}')
