import pandas as pd, numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 x=get_stock_daily_data(s,2200)
 if x is None or len(x)<150:x=get_index_daily_data(s,2200)
 return x.set_index('date').close.astype(float)
p=pd.DataFrame({s:g(s) for s in U}).sort_index();r=p.pct_change(); sig=r.where(r<0,0).rolling(60,min_periods=45).std().shift(1); f=p.shift(-10)/p-1
z=[]; tu=[]; cov=[]
for d in sig.index:
 m=sig.loc[d].notna()&f.loc[d].notna()
 if m.sum()>=8:z.append(spearmanr(sig.loc[d,m],f.loc[d,m]).statistic)
 cov.append(m.mean())
 prev=sig.shift(1).loc[d]; mm=sig.loc[d].notna()&prev.notna()
 if mm.sum()>=8:tu.append((sig.loc[d,mm].rank(pct=True)-prev[mm].rank(pct=True)).abs().mean())
print('dates',len(z),'avg_n',np.mean([((sig.loc[d].notna()&f.loc[d].notna()).sum()) for d in sig.index if (sig.loc[d].notna()&f.loc[d].notna()).sum()>=8]),'coverage',np.mean(cov),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z)*np.sqrt(252),'hit',np.mean(np.array(z)>0),'turnover',np.mean(tu))
for h in [1,5,20]:
 ff=p.shift(-h)/p-1; q=[]
 for d in sig.index:
  m=sig.loc[d].notna()&ff.loc[d].notna()
  if m.sum()>=8:q.append(spearmanr(sig.loc[d,m],ff.loc[d,m]).statistic)
 print(h,np.mean(q),np.mean(q)/np.std(q)*np.sqrt(252),len(q))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20270826_downside_stress_signal.csv',index=False)
