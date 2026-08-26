import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-01-10')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index();P=P[P.index<=end]; R=P.pct_change(); vol=R.rolling(20).std(); shock=-(P/P.shift(3)-1)/(vol*np.sqrt(3)+1e-9)
med=R.median(axis=1); disp=R.sub(med,axis=0).abs().median(axis=1)
gate=(disp>disp.rolling(80).median()).astype(float); f=shock.mul(gate,axis=0).clip(-8,8)
for h in [1,5,10]:
 z=[]; ns=[]
 for dt in f.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; ok=f.loc[dt].notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt][ok],y[ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum())
 a=np.array(z);print('h',h,'n',len(a),'N',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
r=f.rank(axis=1,pct=True);print('turn',((r-r.shift()).abs().mean(axis=1)).dropna().mean())
f.to_csv('scripts/miner_1_20310113_disp3_signal.csv');pd.DataFrame({'ic':z}).to_csv('scripts/miner_1_20310113_disp3_ic.csv',index=False)
