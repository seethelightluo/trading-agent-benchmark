import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-03-24')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:end] for s in U};p=pd.DataFrame(px);r=p.pct_change()
# low realized volatility, modestly smoothed by recent downside asymmetry
sig=-r.rolling(20).std()
def ev(h):
 y=p.pct_change(h).shift(-h); a=[]; ns=[]; dates=[]
 for t in sig.index:
  x=sig.loc[t];z=y.loc[t];ok=x.notna()&z.notna()
  if ok.sum()>=8:a.append(spearmanr(x[ok],z[ok]).statistic);ns.append(ok.sum());dates.append(t)
 a=np.array(a);return len(a),np.mean(a),np.mean(a)/np.std(a),np.mean(a>0),np.mean(ns),dates,a
for h in [1,5,10]:
 q=ev(h);print(h,'dates',q[0],'IC',q[1],'ICIR',q[2],'hit',q[3],'N',q[4])
q=ev(1);a=q[6];
for aa,bb in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-24')]:
 vals=[v for d,v in zip(q[5],a) if pd.Timestamp(aa)<=d<=pd.Timestamp(bb)];print(aa,len(vals),np.mean(vals) if vals else np.nan,np.mean(vals)/np.std(vals) if len(vals)>1 else np.nan)
print('coverage',sig.notna().sum().sum()/(sig.shape[0]*15),'turn',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
sig.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20270325_lowvol_signal.csv',index=False)
