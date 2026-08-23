import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-04-18'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
r5=P/P.shift(5)-1
# Short reversal, strengthened when cross-sectional dispersion is high (shock regimes).
csdisp=R.rolling(20,min_periods=15).std().mean(axis=1)
scale=csdisp/(csdisp.rolling(120,min_periods=40).median()+1e-12)
F=(-r5/(vol+1e-12)).mul(1+0.75*(scale-1).clip(-1,1),axis=0)
rows=[]
for i in range(140,len(P)-10):
 a=F.iloc[i].values; b=(P.iloc[i+10]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
print(f'data_dates {len(P)} instruments {len(U)} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {ic:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('turnover_proxy', (F.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.10).mean())
