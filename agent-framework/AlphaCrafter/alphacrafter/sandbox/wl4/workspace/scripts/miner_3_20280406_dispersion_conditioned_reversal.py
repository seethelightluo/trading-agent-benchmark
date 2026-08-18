import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'),usecols=['date','close']); d['date']=pd.to_datetime(d.date); px[s]=d.set_index('date').close
prices=pd.DataFrame(px).sort_index(); r=prices.pct_change()
disp=r.shift(1).rolling(10).apply(lambda x: np.std(x), raw=True)
# rolling apply above is per column; use cross-sectional dispersion directly each date
csdisp=r.shift(1).apply(lambda x: x.std(),axis=1)
disp_z=(csdisp-csdisp.rolling(120).mean())/(csdisp.rolling(120).std()+1e-12)
raw=-(prices.shift(1)/prices.shift(11)-1)/(r.shift(1).rolling(20).std()*np.sqrt(20))
f=raw.mul((1+0.5*np.tanh(disp_z)),axis=0); f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; vals=[]; dates=[]; counts=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(a[ok],b[ok]).statistic); dates.append(dt); counts.append(ok.sum())
 x=pd.Series(vals,index=dates).dropna(); print(f'h={h} dates={len(x)} avg_n={np.mean(counts):.2f} coverage={np.mean(counts)/15:.4f} IC={x.mean():.6f} ICIR={x.mean()/x.std():.6f} hit={np.mean(x>0):.4f}')
 for label,z in [('early',x.iloc[:len(x)//2]),('late',x.iloc[len(x)//2:])]: print(' ',label,len(z),f'IC={z.mean():.6f}',f'ICIR={z.mean()/z.std():.6f}',f'hit={np.mean(z>0):.4f}')
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'n_dates',len(prices),'n_assets',prices.shape[1])
