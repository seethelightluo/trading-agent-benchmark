import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-12'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
# Low-turnover multi-horizon relative reversal: contrarian 3d shock, gated by 20d volatility.
r3=P.pct_change(3); r10=P.pct_change(10); vol20=R.rolling(20).std()
rel3=r3.sub(r3.median(axis=1),axis=0); rel10=r10.sub(r10.median(axis=1),axis=0)
f=-(0.7*rel3+0.3*rel10)/(vol20.clip(lower=1e-6))
y=P.shift(-10)/P-1
def calc(x, sl=slice(None)):
 a=[]; ns=[]
 for dt in x.loc[sl].index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
for name,x in [('candidate',f),('plain3',-rel3),('plain10',-rel10)]:
 print(name,'dates avgN IC ICIR hit',tuple(round(v,6) if isinstance(v,float) else v for v in calc(x)))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-12')]:
  q=calc(x,slice(lo,hi)); print('REG',lo,hi,'dates',q[0],'N',round(q[1],2),'IC',round(q[2],6),'ICIR',round(q[3],6))
rk=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round((rk-rk.shift()).abs().mean(axis=1).dropna().mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20280113_multihorizon_volneutral_signal.csv')
