import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-03-21'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); rv=R.rolling(20,min_periods=15).std()*np.sqrt(20)
r5=P/P.shift(5)-1; r20=P/P.shift(20)-1
# Short-term reversal, strengthened when the asset's medium-term trend is weak;
# cross-sectional factor is lagged by one completed close through the construction date.
market=R.mean(axis=1); dispersion=R.std(axis=1).rolling(20,min_periods=15).mean()
reg=(dispersion/(dispersion.rolling(120,min_periods=40).median()+1e-12)).clip(0.5,2.0)
F=(-r5/(rv+1e-12))*(1+0.5*(1-(r20.abs()/(r20.abs().rolling(120,min_periods=40).median()+1e-12)).clip(0,2)))*reg.values[:,None]
def run(h):
 rows=[]
 for i in range(140,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(rows,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} data_dates {len(P)} instruments {len(U)} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {ic:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f} turnover {F.iloc[1:].rank(axis=1,pct=True).diff().abs().mean().mean():.5f}')
 if h==10: print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
for h in [5,10,20]: run(h)
