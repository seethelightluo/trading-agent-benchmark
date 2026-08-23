import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-09-19')
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index().dropna();r=p.pct_change(); dn=r.clip(upper=0).abs().rolling(20,min_periods=15).mean(); base=(-r.rolling(3).sum()/dn.clip(lower=1e-5)).rank(axis=1,pct=True)
# cross-asset regime: median volatility relative to its trailing 60d median
vol=r.rolling(20,min_periods=15).std(); mvol=vol.median(axis=1); med=mvol.rolling(60,min_periods=40).median(); ratio=mvol/med
variants={'base':base,'lowvol_gate':base.mul((ratio<1).astype(float),axis=0),'highvol_gate':base.mul((ratio>=1).astype(float),axis=0),'smooth_lowvol':base.mul((2-ratio).clip(0.25,1.75),axis=0)}
def calc(sig,h,a='2020-01-01',b='2029-09-19'):
 z=[];nn=[];ds=[]
 for i in range(len(p)-h):
  if not(pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(q.f.corr(q.y,method='spearman'));nn.append(len(q));ds.append(p.index[i])
 x=pd.Series(z,index=ds).dropna();return len(x),np.mean(nn),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
print('rows',len(p),'assets',p.shape[1])
for name,s in variants.items():
 for h in [5,10,20]: print(name,h,calc(s,h))
for name,s in variants.items():
 print('REGIME',name,'10d')
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-09-19')]:print(a,calc(s,10,a,b))
out=variants['smooth_lowvol'].reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20290920_volregime_downside_reversal_signal.csv',index=False);print('artifact',len(out),'dates',out.date.nunique(),'assets',out.symbol.nunique())
