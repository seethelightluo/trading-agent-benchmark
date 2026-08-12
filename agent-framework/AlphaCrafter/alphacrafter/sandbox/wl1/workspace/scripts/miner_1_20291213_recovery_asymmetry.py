import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3600) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index().ffill()
r=px.pct_change()
# recovery from 60d trough, conditioned on positive 20d slope and penalized downside risk
low=px.rolling(60,min_periods=40).min()
recovery=px/low-1
mom=px.pct_change(20)
down=r.where(r<0).rolling(40,min_periods=25).std()
f=(recovery * (1+mom.clip(lower=-.5)) / (down*np.sqrt(252))).shift(1)
# winsorize cross section
f=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1),axis=0)
for h in [1,5,10,20]:
  fr=px.pct_change(h).shift(-h)
  vals=[]; dates=[]; n=[]
  for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
      vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt);n.append(len(z))
  x=pd.Series(vals,index=dates).dropna()
  print('h',h,'dates',len(x),'avgN',np.mean(n),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean()))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
# regime 2020-25, 2026+, 2028+, 2029
fr=px.pct_change(20).shift(-20); x=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:x.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.Series(dict(x));
for a,b in [('2020','2025-12-31'),('2026','2027-12-31'),('2028','2028-12-31'),('2029','2029-12-12')]:
 q=x.loc[a:b];print(a,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20291213_recovery_asymmetry_signal.csv')
