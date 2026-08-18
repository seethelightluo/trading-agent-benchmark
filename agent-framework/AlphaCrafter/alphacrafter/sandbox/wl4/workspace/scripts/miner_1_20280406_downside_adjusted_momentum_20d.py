import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),usecols=['date','close']); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close
prices=pd.DataFrame(px).sort_index()
# Lagged 20-day return divided by lagged downside deviation; cross-sectional demean.
r=prices.pct_change(); mom=prices.shift(1)/prices.shift(21)-1
neg=r.where(r<0,0.0).rolling(20).std().shift(1)*np.sqrt(20)
f=(mom/neg.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; vals=[]; counts=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic); counts.append(ok.sum())
 x=pd.Series(vals).dropna()
 print(f'h={h} dates={len(x)} avg_n={np.mean(counts):.2f} coverage={np.mean(counts)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std():.6f} hit={np.mean(x>0):.4f}')
 for label,z in [('early',x.iloc[:len(x)//2]),('late',x.iloc[len(x)//2:])]: print(' ',label,len(z),f'IC={z.mean():.6f}',f'ICIR={z.mean()/z.std():.6f}',f'hit={np.mean(z>0):.4f}')
rank=f.rank(axis=1,pct=True); print(f'turnover={rank.diff().abs().mean(axis=1).dropna().mean():.6f} n_dates={len(prices)} n_assets={prices.shape[1]}')
