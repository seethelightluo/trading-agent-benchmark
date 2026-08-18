import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
px={a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for a in A if os.path.exists(f'{b}/{a}.csv')}
P=pd.DataFrame(px).sort_index(); P=P.loc[:'2028-07-12']; ret=P.pct_change();
# Trend agreement: directionally consistent multi-horizon momentum, risk scaled and cross-sectionally centered.
mom20=P.pct_change(20); vol=ret.rolling(20).std(); base=mom20/(vol*np.sqrt(20)+1e-12)
agree=((P.pct_change(5)>0).astype(float)+(P.pct_change(10)>0).astype(float)+(P.pct_change(20)>0).astype(float))/3-.5
sig=base*agree
sig=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1)+1e-12,axis=0).shift(1)
def ic(x,y):
 ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<8:return np.nan,ok.sum()
 return np.corrcoef(pd.Series(x[ok]).rank(),pd.Series(y[ok]).rank())[0,1],ok.sum()
print('rows',len(P),'assets',len(px),'end',P.index.max().date())
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  q,n=ic(sig.iloc[i].to_numpy(),(P.iloc[i+h]/P.iloc[i]-1).to_numpy())
  if np.isfinite(q): vals.append(q);ns.append(n)
 s=pd.Series(vals); r=s.tail(250)
 print('h',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(r.mean(),6),round(r.mean()/r.std(ddof=1),6))
rank=sig.rank(axis=1,pct=True); print('coverage',round(sig.notna().sum().sum()/(len(sig)*len(px)),4),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
for label,start,end in [('early','2020','2023-12-31'),('mid','2024','2026-12-31'),('late','2027','2028-07-12')]:
 vals=[]
 for i,d in enumerate(P.index):
  if pd.Timestamp(start)<=d<=pd.Timestamp(end) and i+20<len(P):
   q,n=ic(sig.iloc[i].to_numpy(),(P.iloc[i+20]/P.iloc[i]-1).to_numpy())
   if np.isfinite(q):vals.append(q)
 print('regime',label,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None,'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6) if len(vals)>1 else None)
