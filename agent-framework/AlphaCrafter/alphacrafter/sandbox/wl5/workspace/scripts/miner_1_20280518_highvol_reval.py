import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-18')
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close; p[a]=d[d.index<=end]
pd_=pd.DataFrame(p).sort_index(); r=pd_.pct_change(); v=r.rolling(20,min_periods=15).std(); rr=r.rolling(5,min_periods=5).sum(); f=.8*v.rank(axis=1,pct=True)-.2*rr.rank(axis=1,pct=True)
rows=[]; sig=[]
for i in range(len(pd_)-10):
 z=pd.concat([f.iloc[i],pd_.iloc[i+10]/pd_.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((pd_.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))); sig.append((pd_.index[i],*f.iloc[i].values))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'range',q.index.min(),q.index.max(),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
for name,sub in [('2020-24',q.loc['2020':'2024']),('2025-26',q.loc['2025':'2026']),('2027-28',q.loc['2027':'2028'])]: print(name,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std() if len(sub)>1 else np.nan)
s=pd.DataFrame([x[1:] for x in sig],index=[x[0] for x in sig],columns=A); print('coverage',q.n.mean()/15,'turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()); s.to_csv('scripts/miner_1_20280518_highvol_reval_signal.csv')
