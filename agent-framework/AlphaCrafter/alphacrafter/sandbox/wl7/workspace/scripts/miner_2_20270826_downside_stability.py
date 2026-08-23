import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    x=get_stock_daily_data(s,2200)
    if x is None or len(x)<150: x=get_index_daily_data(s,2200)
    return x.set_index('date')['close'].astype(float) if x is not None else pd.Series(dtype=float)
px=pd.DataFrame({s:get(s) for s in U}).sort_index()
r=px.pct_change()
# lagged downside-tail stability: reward assets with low downside deviation, mildly favor positive drift
# all inputs at t-1, forward return t+1..t+10
sig=(-r.where(r<0,0).rolling(60,min_periods=45).std()).shift(1)
fwd=px.shift(-10)/px-1
rows=[]; turnover=[]; cov=[]
for d in sig.index:
    a=sig.loc[d]; y=fwd.loc[d]; m=a.notna()&y.notna()
    if m.sum()>=8:
        rows.append((d,spearmanr(a[m],y[m]).statistic,m.sum()))
    prev=sig.shift(1).loc[d]; mm=a.notna()&prev.notna()
    if mm.sum()>=8:
        turnover.append((a[mm].rank(pct=True)-prev[mm].rank(pct=True)).abs().mean())
    cov.append(m.mean())
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',np.mean(cov),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std()*np.sqrt(252),'hit',np.mean(q.ic>0),'turnover',np.mean(turnover))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; z=[]
 for d in sig.index:
  m=sig.loc[d].notna()&yy.loc[d].notna()
  if m.sum()>=8:z.append(spearmanr(sig.loc[d,m],yy.loc[d,m]).statistic)
 print(h,np.mean(z),np.mean(z)/np.std(z)*np.sqrt(252),len(z))
for name,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-27':('2025','2027-12-31')}.items():
 z=q.loc[a:b,'ic'];print(name,len(z),z.mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20270826_downside_stability_signal.csv',index=False)
