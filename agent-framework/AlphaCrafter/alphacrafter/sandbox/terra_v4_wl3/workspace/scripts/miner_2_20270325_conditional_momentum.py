import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-03-25'); C={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); C[s]=d.loc[d.index<=cut,'close']
p=pd.DataFrame(C).sort_index(); r=p.pct_change();
# momentum persistence: 3d return, scaled by 20d vol, with cross-sectional market trend conditioning
f=(p.shift(1)/p.shift(4)-1)/r.rolling(20,min_periods=15).std().shift(1)
# conditional: multiply by sign of median 20d market return, aiming trend in risk-on and reversal risk-off
reg=(p.pct_change(20).median(axis=1)>0).astype(float).replace(0,-1).shift(1)
f=f.mul(reg,axis=0).rank(axis=1,pct=True)
for h in [1,5,10]:
 y=p.shift(-h)/p-1;a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.median(ns))
print('coverage',f.notna().mean().mean(),'turn',f.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270325_conditional_momentum_signal.csv',index=False)
