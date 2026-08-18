import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'))
 d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change()
# lagged multi-horizon trend quality: agreement of medium and long trends, volatility scaled
m20=P.pct_change(20); m60=P.pct_change(60); vol=P.pct_change().rolling(20).std()*np.sqrt(20)
agreement=np.where(np.sign(m20)==np.sign(m60),1.0,0.35)
F=((0.6*m20+0.4*m60)/(vol.replace(0,np.nan)))*agreement
# enforce one day information lag
F=F.shift(1)
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(P)-h):
  f=F.iloc[i]; y=P.pct_change(h).iloc[i+h]
  z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); ic=np.nanmean(a); ir=ic/np.nanstd(a,ddof=1)*np.sqrt(len(a)) if len(a)>1 else np.nan
 print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
# daily signal rank turnover, coverage, regimes
rank=F.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).mean()
print(f'coverage={F.notna().mean().mean():.6f} turnover={turn:.6f} dates={len(P)} instruments={len(U)}')
# 10d IC by broad calendar regimes
vals=[]
for i in range(len(P)-10):
 z=pd.concat([F.iloc[i],P.pct_change(10).iloc[i+10]],axis=1).dropna()
 if len(z)>=8: vals.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(vals,columns=['date','ic']); df['date']=pd.to_datetime(df.date)
for a,b in [('2020','2022'),('2022','2024'),('2024','2026'),('2026','2028'),('2028','2030')]:
 x=df[(df.date>=a)&(df.date<b)].ic
 print('regime',a,b,'n',len(x),'ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(len(x)) if len(x)>1 else np.nan)
