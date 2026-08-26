import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-10-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:END]
r=px.pct_change(); ret20=px.pct_change(20); path=r.abs().rolling(20).sum(); f=ret20/path
# efficiency ratio: directional displacement over realized path, interpretable trend persistence
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in px.index:
  a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic); ds.append(dt); ns.append(ok.sum())
 z=pd.Series(vals,index=ds).dropna(); print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),len(z),np.mean(ns[:len(z)]),(z>0).mean()))
q=pd.Series(vals,index=ds).dropna()
for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 x=q.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(len(x)),len(x),(x>0).mean()))
# rank turnover and coverage
rank=f.rank(axis=1,pct=True); print('COVERAGE',f.notna().mean().mean(),'TURNOVER',rank.diff().abs().mean(axis=1).dropna().mean(),'LAST',px.index[-1].date())
# save reproducible signal artifact
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_3_20341016_efficiency20_signal.csv',index=False)
