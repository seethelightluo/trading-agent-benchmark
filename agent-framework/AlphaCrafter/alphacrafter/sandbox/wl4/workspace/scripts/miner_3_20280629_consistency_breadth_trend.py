import pandas as pd, numpy as np, os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'; px={}
for a in A:
 path=f'{b}/{a}.csv'
 if os.path.exists(path): px[a]=pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index()['close']
P=pd.DataFrame(px).sort_index(); P=P.loc[:'2028-06-29']; R=P.pct_change(); r20=P.pct_change(20)
# Trend consistency: fraction of positive 5-day subperiod returns over trailing 20d, centered at 0.5;
# amplify cross-sectional 20d risk-adjusted trend only when broad participation confirms it.
sub=[P.pct_change(k).shift(0) for k in [5,10,15,20]]
cons=sum((x>0).astype(float) for x in sub)/4
vol=R.rolling(20).std(); mom=r20/(vol*np.sqrt(20)+1e-12)
breadth=(r20>0).sum(axis=1)/r20.notna().sum(axis=1)
confirm=2*(breadth-.5)
z=mom.sub(mom.mean(axis=1),axis=0).div(mom.std(axis=1)+1e-12,axis=0)
signal=(z*(cons-.5)*2).mul((0.5+0.5*confirm.abs()),axis=0).shift(1)
def rc(x,y):
 ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<8:return np.nan,ok.sum()
 return np.corrcoef(pd.Series(x[ok]).rank(),pd.Series(y[ok]).rank())[0,1],ok.sum()
print('shapes',P.shape,mom.shape,cons.shape,signal.shape); print('assets',len(px),'rows',len(P),'date_end',P.index.max().date())
for h in [1,5,10,20]:
 y=(P.shift(-h)/P-1).to_numpy(); out=[rc(signal.iloc[i].to_numpy(),y[i]) for i in range(len(P))]; zc=np.array([q[0] for q in out if np.isfinite(q[0])]); ns=np.array([q[1] for q in out if np.isfinite(q[0])]); s=pd.Series(zc); recent=s.tail(250).dropna()
 print('horizon',h,'dates',len(s),'avg_n',round(ns.mean(),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
rank=signal.rank(axis=1,pct=True); print('coverage',round(signal.notna().sum().sum()/(len(signal)*len(px)),4),'turnover_mean',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
for label,start,end in [('early','2020','2023-12-31'),('mid','2024','2026-12-31'),('late','2027','2028-06-29')]:
 vals=[]
 for i,d in enumerate(P.index):
  if d>=pd.Timestamp(start) and d<=pd.Timestamp(end):
   q=rc(signal.iloc[i].to_numpy(),((P.shift(-20)/P-1).iloc[i]).to_numpy())[0]
   if np.isfinite(q): vals.append(q)
 print('regime',label,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None,'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6) if len(vals)>1 else None)
