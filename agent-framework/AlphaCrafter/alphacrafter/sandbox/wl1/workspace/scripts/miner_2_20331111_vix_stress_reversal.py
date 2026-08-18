import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-11-11')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
pd_=pd.DataFrame(p).sort_index().loc[:end]; r=pd_.pct_change()
v=r.rolling(20).std()*np.sqrt(252); rev=-pd_.pct_change(10); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(pd_.index).ffill(); vp=vix.rolling(120,min_periods=60).rank(pct=True)
f=rev.div(v).mul(vp,axis=0); fr=pd_.shift(-10).div(pd_)-1
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(c): rows.append((d,c,len(z)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(df),'avg_n',df.n.mean(),'coverage',df.n.mean()/15,'IC',df.ic.mean(),'ICIR',df.ic.mean()/df.ic.std(ddof=1),'hit',(df.ic>0).mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=df.loc[a:b]; print(a+'-'+b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
rank=f.rank(axis=1,pct=True); print('turnover',(rank-rank.shift(10)).abs().mean(axis=1).dropna().mean(),'latest',df.index[-1].date())
for h in [5,10,20]:
 rr=pd_.shift(-h).div(pd_)-1; vals=[]
 for d in f.index:
  z=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(c): vals.append(c)
 print('h',h,'IC',np.mean(vals),'n',len(vals))
pd.DataFrame(f,index=pd_.index,columns=U).to_csv('scripts/miner_2_20331111_vix_stress_reversal_signal.csv',index_label='date')
