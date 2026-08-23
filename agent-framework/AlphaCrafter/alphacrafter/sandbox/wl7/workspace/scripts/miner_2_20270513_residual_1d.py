import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-05-12')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(px).sort_index(); p=p[p.index<=end]; r=p.pct_change(1)
# residual one-day return reversal, lagged
s=-(r.sub(r.mean(axis=1),axis=0)).shift(1); out=[]
for d in s.index:
 z=pd.concat([s.loc[d],r.shift(-1).loc[d]],axis=1).dropna()
 if len(z)>=8: out.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(out,columns=['date','n','ic']).set_index('date'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/len(a)/15); print('IC %.8f ICIR %.8f hit %.4f'% (a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean()));
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=a.loc[lo:hi].ic; print(lo,len(q),q.mean(),q.mean()/q.std())
s.to_csv('scripts/miner_2_20270513_residual_1d_signal.csv')
