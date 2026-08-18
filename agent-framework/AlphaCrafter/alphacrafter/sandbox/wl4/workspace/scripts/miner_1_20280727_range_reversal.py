import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
D={a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in A if os.path.exists(f'{b}/{a}.csv')}
P=pd.DataFrame({a:d.close for a,d in D.items()}).sort_index().loc[:'2028-07-26']; r=P.pct_change();
# Range-normalized one-day reversal: fade the prior move, scaled by recent realized risk.
vol=r.rolling(20).std(); sig=-(r/(vol+1e-12)); sig=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1)+1e-12,axis=0).shift(1)
def ic(x,y):
 ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<8:return np.nan,ok.sum()
 return np.corrcoef(pd.Series(x[ok]).rank(),pd.Series(y[ok]).rank())[0,1],ok.sum()
print('rows',len(P),'assets',len(D),'end',P.index.max().date())
for h in [1,5,10,20]:
 v=[];n=[]
 for i in range(len(P)-h):
  q,k=ic(sig.iloc[i].values,(P.iloc[i+h]/P.iloc[i]-1).values)
  if np.isfinite(q):v.append(q);n.append(k)
 s=pd.Series(v);z=s.tail(250)
 print('h',h,'dates',len(s),'avg_n',round(np.mean(n),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
rank=sig.rank(axis=1,pct=True); print('coverage',round(sig.notna().sum().sum()/(len(sig)*len(D)),4),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
for lab,st,en in [('early','2020','2023-12-31'),('mid','2024','2026-12-31'),('late','2027','2028-07-26')]:
 v=[]
 for i,d in enumerate(P.index):
  if pd.Timestamp(st)<=d<=pd.Timestamp(en) and i+1<len(P):
   q,k=ic(sig.iloc[i].values,(P.iloc[i+1]/P.iloc[i]-1).values)
   if np.isfinite(q):v.append(q)
 print('regime',lab,'dates',len(v),'IC',round(np.mean(v),6),'ICIR',round(np.mean(v)/np.std(v,ddof=1),6))
