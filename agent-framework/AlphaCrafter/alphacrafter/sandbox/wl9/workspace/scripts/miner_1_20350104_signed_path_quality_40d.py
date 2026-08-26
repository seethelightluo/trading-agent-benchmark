import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); D[s]=d[d.date<='2035-01-03'].set_index('date')['close']
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); w=40
ret=px.pct_change(w); cons=r.rolling(w).mean().abs()/(r.abs().rolling(w).mean()+1e-12); sig=ret*cons
for h in [5,10,20,40]:
 out=[]
 for i in range(w,len(px)-h):
  z=pd.concat([sig.iloc[i],(px.iloc[i+h]/px.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(out); print('H',h,'dates',len(a),'meanIC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(252),'hit',np.mean(a>0))
print('period',px.index.min().date(),px.index.max().date(),'avgN',px.notna().sum(axis=1).mean(),'coverage',px.notna().mean().mean())
ranks=sig.rank(axis=1,pct=True); print('turnover',(ranks.diff().abs().mean(axis=1)/2).mean())
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2032','2035')]:
 out=[]
 for i in range(w,len(px)-10):
  dt=px.index[i]
  if not (pd.Timestamp(a+'-01-01')<=dt<=pd.Timestamp(b+'-12-31')): continue
  z=pd.concat([sig.iloc[i],(px.iloc[i+10]/px.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=np.array(out); print('REG',a,b,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(252))
