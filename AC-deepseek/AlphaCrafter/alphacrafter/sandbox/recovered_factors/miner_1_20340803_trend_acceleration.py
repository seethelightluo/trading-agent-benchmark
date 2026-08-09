import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 q='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(q): P[a]=pd.read_csv(q,parse_dates=['date']).set_index('date')['close']
pd_=pd.DataFrame(P).sort_index(); r=np.log(pd_).diff()
# Trend acceleration: recent 5-session return versus the average daily 20-session return.
# Positive means strengthening trend; lagged one completed session.
r5=r.rolling(5,min_periods=4).sum(); r20=r.rolling(20,min_periods=15).sum()
f=(r5-r20/4).shift(1)
# cross-sectional standardization is not needed for Spearman, but center for clarity
f=f.sub(f.mean(axis=1),axis=0)
print('DATA',pd_.index.min(),pd_.index.max(),'assets',len(P),'rows',len(pd_))
for h in [1,5,10,20]:
 y=pd_.shift(-h)/pd_-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(out); print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
print('coverage',f.notna().stack().mean(),'mean_valid',f.notna().sum(axis=1).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
y=pd_.shift(-1)/pd_-1
for label,sub in [('2020-23',f.loc['2020':'2023']),('2024-27',f.loc['2024':'2027']),('2028-30',f.loc['2028':'2030']),('2031-34',f.loc['2031':'2034'])]:
 x=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(x); print(label,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6))
