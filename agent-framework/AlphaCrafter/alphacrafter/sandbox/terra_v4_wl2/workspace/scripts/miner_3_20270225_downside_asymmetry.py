import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r={a:p[a].pct_change() for a in A}
# Downside asymmetry: recent upside-vs-downside realized volatility, scaled by total vol.
# Positive values indicate returns dominated by positive observations rather than drawdown shocks.
raw={}
for a in A:
 x=r[a]
 down=x.where(x<0,0).rolling(30).std(); up=x.where(x>0,0).rolling(30).std(); tv=x.rolling(30).std()
 raw[a]=(up-down)/(tv+1e-8)
idx=sorted(set().union(*[set(x.index) for x in p.values()])); out=[]; sig=[]
for d in idx:
 v={a:raw[a].get(d,np.nan) for a in A}; good=[z for z in v.values() if np.isfinite(z)]; med=np.nanmedian(good) if len(good)>=8 else np.nan
 for a in A: sig.append((d,a,v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in p[a].index: continue
   i=p[a].index.get_loc(d); z=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
   if i+h<len(p[a]) and np.isfinite(z): f.append(z); y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: out.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(out,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_3_20270225_downside_asymmetry.csv',index=False)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
