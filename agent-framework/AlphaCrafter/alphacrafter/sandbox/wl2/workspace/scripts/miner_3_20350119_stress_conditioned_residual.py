import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); r=P.pct_change(); db=r[D].mean(axis=1)
beta=r.rolling(90,min_periods=60).cov(db).div(db.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r.sub(beta.mul(db,axis=0)); rv=res.rolling(60,min_periods=40).std().shift(1)
base=-res.rolling(30,min_periods=20).sum().shift(1)/(rv*np.sqrt(30)+1e-9)
# Stress-conditioned: apply residual reversal only when defensive basket has positive lagged 60d return.
stress=(db.rolling(60,min_periods=40).sum().shift(1)>0)
sig=base.where(stress)
y=P.pct_change(40).shift(-40)
vals=[]; ns=[]; ds=[]
for d in sig.index:
 ok=sig.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: vals.append(spearmanr(sig.loc[d][ok],y.loc[d][ok]).statistic); ns.append(ok.sum());ds.append(d)
q=pd.Series(vals,index=ds)
print('candidate stress_conditioned_residual_reversal_40d')
print('dates',len(q),'avg_n',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for name,s in [('2026-2030',q.loc['2026':'2030']),('2031-2035',q.loc['2031':'2035']),('2034-2035',q.loc['2034':'2035'])]: print(name,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage %.4f turnover %.4f stress_days %.4f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),stress.mean()))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20350119_stress_conditioned_residual_reversal_signal.csv',index=False)
