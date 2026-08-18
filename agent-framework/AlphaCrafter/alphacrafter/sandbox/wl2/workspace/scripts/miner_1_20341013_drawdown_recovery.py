import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,usecols=['date','close'],parse_dates=['date']).set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Relative drawdown recovery: position above the trailing 120-day low, normalized by recent risk.
low=P.rolling(120,min_periods=60).min()
vol=R.rolling(20,min_periods=10).std()
f=((P/low-1)/(vol*np.sqrt(252)+1e-8)).shift(1)
rows=[]
for h in [1,5,10,20]:
 y=P.pct_change(h).shift(-h); q=[]
 for dt in f.index:
  x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append((dt,spearmanr(x[ok],z[ok]).statistic,ok.sum()))
 d=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); rows.append(d)
 print('h',h,'dates',len(d),'avg_n',d.n.mean(),'IC %.6f ICIR %.6f hit %.3f'%(d.ic.mean(),d.ic.mean()/d.ic.std(ddof=1),(d.ic>0).mean()))
 for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-12-31')]:
  s=d.loc[lo:hi].ic; print(' ',lo, len(s), '%.6f %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_1_20341013_drawdown_recovery_signal.csv',index=False)
