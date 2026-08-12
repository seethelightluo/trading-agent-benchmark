import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close']
prices=pd.DataFrame(px).sort_index(); prices=prices.loc[:'2031-02-19']; r=prices.pct_change()
ret20=prices.pct_change(20); vol40=r.rolling(40).std(); breadth40=(r>0).rolling(40).mean(); bench=ret20.mean(axis=1)
marketbreadth=(r>0).mean(axis=1).rolling(20).mean()
factor=(ret20.sub(bench,axis=0)/(vol40+1e-8))*(0.5+breadth40)*(0.5+marketbreadth.values[:,None]); factor=factor.shift(1)
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; vals=[]; dates=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 x=pd.Series(vals,index=dates).dropna(); print(f'h={h} dates={len(x)} avgN={np.mean(ns):.3f} IC={x.mean():.8f} ICIR={x.mean()/x.std(ddof=1):.8f} hit={(x>0).mean():.4f}')
print('coverage',factor.notna().mean().mean(),'turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean(),'range',prices.index.min(),prices.index.max())
