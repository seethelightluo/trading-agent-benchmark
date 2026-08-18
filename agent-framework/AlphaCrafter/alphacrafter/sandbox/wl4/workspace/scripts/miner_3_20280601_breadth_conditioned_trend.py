import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); n=len(px); ret20=P.pct_change(20); breadth=(ret20>0).mean(axis=1); strength=(breadth-.5)*2
f=ret20.sub(ret20.mean(axis=1),axis=0).div(ret20.std(axis=1)+1e-12,axis=0).mul(strength,axis=0).shift(1)
def rc(x,y):
 ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
 if len(x)<8:return np.nan,len(x)
 return np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1],len(x)
for h in [1,5,10,20]:
 y=(P.shift(-h)/P-1).to_numpy(); X=f.to_numpy(); out=[rc(X[i],y[i]) for i in range(len(P))]; z=np.array([q[0] for q in out if np.isfinite(q[0])]); ns=np.array([q[1] for q in out if np.isfinite(q[0])]); s=pd.Series(z).dropna(); recent=s.tail(250)
 print('horizon',h,'dates',len(s),'avg_n',round(ns.mean(),2),'coverage',round(np.isfinite(X).sum()/(len(X)*n),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
print('assets',n,'rows',len(P),'valid_factor_rate',round(f.notna().sum().sum()/(len(f)*n),4))
