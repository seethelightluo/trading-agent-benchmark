import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); dates=p.index
# Candidate: dispersion-conditioned risk-adjusted relative strength, calculated at t, fwd return t+1
# high dispersion means cross-sectional opportunity; use 10d relative strength / 20d vol, demean cross-section
csdisp=r.std(axis=1).rolling(60).rank(pct=True)
base=(p.pct_change(10) / r.rolling(20).std()).replace([np.inf,-np.inf],np.nan)
f=base.sub(base.mean(axis=1),axis=0).where(csdisp>0.65)
# evaluate
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1
 vals=[]; turnover=[]; ns=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=fr.iloc[i]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 print('H',h,'dates',len(vals),'meanIC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1),'hit',np.mean(np.array(vals)>0),'meanN',np.mean(ns))
# regimes and signal turnover/persistence
v=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append((dates[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
v=pd.DataFrame(v,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-08','2031-01')]:
 q=v.loc[a:b,'ic'];print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'active dates',f.notna().any(axis=1).sum())
# rank turnover on consecutive active dates
rank=f.rank(axis=1,pct=True); print('turn',rank.diff().abs().mean().mean(),'persist',rank.corrwith(rank.shift(1),axis=1).mean())
