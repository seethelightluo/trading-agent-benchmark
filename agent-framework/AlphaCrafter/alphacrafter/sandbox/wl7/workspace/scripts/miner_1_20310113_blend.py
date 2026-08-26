import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-01-10')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index();P=P[P.index<=end];R=P.pct_change(); vol=R.rolling(20).std()
def sh(h): return -(P/P.shift(h)-1)/(vol*np.sqrt(h)+1e-9)
f=(.7*sh(3)+.3*sh(10)); med=R.median(axis=1);disp=R.sub(med,axis=0).abs().median(axis=1); f=f.mul((disp>disp.rolling(80).median()).astype(float),axis=0).clip(-8,8)
for h in [1,5,10]:
 a=[];ns=[]
 y=P.shift(-h)/P-1
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt][ok],y.loc[dt][ok]).statistic
   if np.isfinite(q):a.append(q);ns.append(ok.sum())
 a=np.array(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
print('turn',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).dropna().mean())
f.to_csv('scripts/miner_1_20310113_blend_signal.csv')
