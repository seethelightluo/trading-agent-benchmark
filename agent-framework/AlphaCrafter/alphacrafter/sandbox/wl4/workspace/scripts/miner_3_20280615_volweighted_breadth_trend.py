import pandas as pd, numpy as np, os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r20=P.pct_change(20); v20=R.rolling(20).std()*np.sqrt(252)
# Volatility-weighted participation: high-vol assets receive less influence; lag all inputs before forward return
w=1/(v20+1e-12); pos=(r20>0).astype(float); breadth=(pos*w).sum(axis=1)/w.where(pos.notna()).sum(axis=1)
strength=(breadth-.5)*2
z=r20.sub(r20.mean(axis=1),axis=0).div(r20.std(axis=1)+1e-12,axis=0)
f=z.mul(strength,axis=0).shift(1)
def rc(x,y):
 ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<8:return np.nan,ok.sum()
 return np.corrcoef(pd.Series(x[ok]).rank(),pd.Series(y[ok]).rank())[0,1],ok.sum()
print('assets',len(px),'rows',len(P))
for h in [1,5,10,20]:
 y=(P.shift(-h)/P-1).to_numpy(); X=f.to_numpy(); out=[rc(X[i],y[i]) for i in range(len(P))]; z=np.array([q[0] for q in out if np.isfinite(q[0])]); ns=np.array([q[1] for q in out if np.isfinite(q[0])]); s=pd.Series(z); recent=s.tail(250).dropna();
 print('horizon',h,'dates',len(s.dropna()),'avg_n',round(ns.mean(),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
# signal turnover as rank-order changes, and broad regime split
rank=f.rank(axis=1,pct=True); turnover=(rank.diff().abs().mean(axis=1)).dropna(); print('coverage',round(f.notna().sum().sum()/(len(f)*len(px)),4),'turnover_mean',round(turnover.mean(),6))
for label, sl in [('early',slice('2020','2023-12-31')),('mid',slice('2024','2026-12-31')),('late',slice('2027','2028-05-31'))]:
 vals=[]
 for i in range(len(P)):
  if not (P.index[i]>=pd.Timestamp(sl.start) and P.index[i]<=pd.Timestamp(sl.stop)): continue
  q=rc(f.iloc[i].to_numpy(),((P.shift(-20)/P-1).iloc[i]).to_numpy())[0]
  if np.isfinite(q): vals.append(q)
 print('regime',label,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None,'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6) if len(vals)>1 else None)
