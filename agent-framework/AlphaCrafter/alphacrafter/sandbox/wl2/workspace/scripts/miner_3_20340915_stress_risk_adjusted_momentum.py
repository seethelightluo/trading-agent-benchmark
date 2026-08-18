import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 f=f'{base}/{a}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date'); px[a]=d['close']
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(prices.index).ffill()
# Risk-adjusted medium momentum, conditioned by stress: during rising VIX penalize downside volatility more
r20=prices.pct_change(20)
down=ret.where(ret<0).rolling(20,min_periods=10).std()
vixchg=vix.pct_change(5)
stress=(vixchg>0).astype(float)
# continuous interpretable stress multiplier; only information through t, signal is lagged one day
factor=r20/(down*np.sqrt(252)+1e-8) / (1+0.75*stress.values[:,None]*down.values)
factor=factor.replace([np.inf,-np.inf],np.nan).shift(1)
fwd=prices.pct_change().shift(-1)
rows=[]
for dt in factor.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'mean_n',r.n.mean(),'coverage',r.n.mean()/15)
print('mean_ic %.6f icir %.6f hit %.4f turnover %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), factor.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-09-15')]:
 z=r.loc[lo:hi].ic; print(lo,hi,'n',len(z),'ic %.6f icir %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1), (z>0).mean()))
for h in [1,5,10,20]:
 yy=prices.pct_change(h).shift(-h); rr=[]
 for dt in factor.index:
  x=factor.loc[dt]; y=yy.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(spearmanr(x[ok],y[ok]).statistic)
 rr=pd.Series(rr); print('horizon',h,'ic %.6f icir %.6f'%(rr.mean(),rr.mean()/rr.std(ddof=1)))
# save exact signal artifact for provenance
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('../persistent/miner_3_20340915_stress_risk_adjusted_momentum_signal.csv',index=False)
