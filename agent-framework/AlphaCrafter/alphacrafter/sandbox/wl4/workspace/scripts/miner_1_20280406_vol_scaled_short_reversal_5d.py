import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),usecols=['date','close']); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); ret=p.pct_change(); vol=ret.rolling(20).std().shift(1); f=(-(p.shift(1)/p.shift(6)-1)/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan); f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic); ns.append(ok.sum())
 x=pd.Series(vals).dropna(); print(f'h={h} dates={len(x)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std():.6f} hit={np.mean(x>0):.4f}')
 for lab,z in [('early',x.iloc[:len(x)//2]),('late',x.iloc[len(x)//2:])]: print(' ',lab,len(z),f'IC={z.mean():.6f}',f'ICIR={z.mean()/z.std():.6f}')
print(f'turnover={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean():.6f} n_dates={len(p)} n_assets={p.shape[1]}')
