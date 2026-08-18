import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-20'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change();
r3=R.rolling(3).sum(); r10=R.rolling(10).sum()
# Adaptive horizon contrarian: use fast reversal in high-dispersion days, slower reversal otherwise.
disp=R.sub(R.mean(axis=1),axis=0).abs().mean(axis=1)
threshold=disp.rolling(60,min_periods=30).median()
high=disp>threshold
f_fast=-r3.sub(r3.median(axis=1),axis=0)
f_slow=-r10.sub(r10.median(axis=1),axis=0)
f=f_fast.where(high, f_slow)
y=P.shift(-10)/P-1

def calc(x, sl=slice(None)):
 vals=[]; ns=[]
 for dt in x.loc[sl].index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('adaptive ALL dates N IC ICIR hit',calc(f))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-20')]:
 q=calc(f,slice(lo,hi)); print('REG',lo,hi,'dates',q[0],'N',round(q[1],2),'IC',round(q[2],6),'ICIR',round(q[3],6),'hit',round(q[4],4))
rk=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round((rk-rk.shift()).abs().mean(axis=1).dropna().mean(),4),'high_frac',round(high.mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_3_20280121_adaptive_horizon_signal.csv')
print('artifact=scripts/miner_3_20280121_adaptive_horizon_signal.csv')
