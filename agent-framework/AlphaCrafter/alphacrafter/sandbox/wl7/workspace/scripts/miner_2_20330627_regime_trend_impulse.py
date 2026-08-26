import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-06-26'); prices={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 prices[a]=d['close']
P=pd.DataFrame(prices); r=P.pct_change()
# Short impulse normalized by recent risk, gated by agreement with medium trend.
vol=r.rolling(30,min_periods=20).std(); short=P.pct_change(10); medium=P.pct_change(40)
fac=(short/(vol*np.sqrt(30))).where(np.sign(short)==np.sign(medium),0.0).shift(1)
print('dates',len(fac),'assets',len(assets),'coverage',fac.notna().mean().mean())
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(z))
 x=np.array(vals); thirds=np.array_split(x,3)
 print('H',h,'n',len(x),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f thirds'% (x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)),[round(y.mean(),6) for y in thirds])
# turnover based on rank ordering changes between dates
rank=fac.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
out=[]
for dt in fac.index:
 for a in assets: out.append({'date':dt.date(),'asset':a,'signal':fac.loc[dt,a]})
pd.DataFrame(out).to_csv('scripts/miner_2_20330627_regime_trend_impulse_signal.csv',index=False)
