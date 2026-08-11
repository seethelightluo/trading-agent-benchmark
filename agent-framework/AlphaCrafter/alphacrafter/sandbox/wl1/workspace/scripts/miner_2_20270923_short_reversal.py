import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
end=pd.Timestamp('2027-09-22'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p):p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index()
C=pd.concat({s:load(s).close for s in A},axis=1).loc[:end].sort_index();R=C.pct_change()
# Short-term reversal normalized by recent volatility, lagged one day.
f=(-R.rolling(3).sum()/(R.rolling(20).std()+.005)).shift(1)
rows=[]
for s in A:
 for h in [5,10,20]:
  fw=C[s].shift(-h)/C[s]-1
  z=pd.concat([f[s],fw],axis=1).dropna();
  for dt,(x,y) in z.iterrows():rows.append((dt,s,x,h,y))
df=pd.DataFrame(rows,columns=['date','symbol','factor','h','fwd'])
for h,g in df.groupby('h'):
 a=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8:
   q=spearmanr(x.factor,x.fwd).statistic
   if np.isfinite(q):a.append(q)
 a=np.array(a);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0), 'avgN',len(g)/len(a))
p=df[df.h==20].pivot(index='date',columns='symbol',values='factor');print('coverage',p.notna().mean().mean(),'turnover',p.rank(axis=1,pct=True).diff().abs().mean().mean())
df[df.h==20].to_csv('scripts/miner_2_20270923_short_reversal_signal.csv',index=False)
for lab,m in [('2026+',df.date>='2026-01-01'),('2027',df.date>='2027-01-01'),('Q2+',df.date>='2027-04-01')]:
 a=[]
 for dt,x in df[(df.h==20)&m].groupby('date'):
  if len(x)>=8:a.append(spearmanr(x.factor,x.fwd).statistic)
 a=np.array(a);print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
