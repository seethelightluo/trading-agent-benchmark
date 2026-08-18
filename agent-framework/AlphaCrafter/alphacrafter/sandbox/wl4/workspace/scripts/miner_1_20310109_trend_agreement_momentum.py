import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; end=pd.Timestamp('2031-01-08')
d={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index().loc[:end].ffill()
r=C.pct_change()
# Trend-agreement momentum: medium trend is rewarded when 10/20/60d returns agree;
# disagreement is shrunk, reducing false signals at turning points. Lagged one day.
r10=C.pct_change(10); r20=C.pct_change(20); r60=C.pct_change(60)
agree=(np.sign(r10)+np.sign(r20)+np.sign(r60)).abs()/3.0
raw=(0.25*r10+0.50*r20+0.25*r60)*agree
sig=raw.rank(axis=1,pct=True).shift(1)
rows=[]
h=10
for i in range(len(C)-h):
 z=pd.concat([sig.iloc[i],C.iloc[i+h]/C.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((C.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']); rec=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
def met(q): return (q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(len(q)))
print(f'dates={len(a)} avgN={a.n.mean():.2f} IC={met(a)[0]:.6f} ICIR={met(a)[1]:.6f} hit={(a.ic>0).mean():.4f} recentIC={met(rec)[0]:.6f} recentICIR={met(rec)[1]:.6f}')
print(f'coverage={sig.notna().mean().mean():.4f} turnover={sig.diff().abs().mean(axis=1).mean():.6f} instruments={len(U)} cutoff={C.index[-1].date()}')
for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2027-12-31'),('2028-01-01','2031-01-08')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; print(f'regime {lo}:{hi} dates={len(q)} IC={q.ic.mean():.6f} hit={(q.ic>0).mean():.4f}')
