import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
    px[s]=d
# common dates and completed through prior trading date
close=pd.concat({s: px[s]['close'] for s in U},axis=1).sort_index()
high=pd.concat({s:px[s]['high'] for s in U},axis=1).sort_index(); low=pd.concat({s:px[s]['low'] for s in U},axis=1).sort_index()
cut=pd.Timestamp('2035-05-27'); close=close.loc[:cut]; high=high.loc[:cut]; low=low.loc[:cut]
# close-location pressure: average CLV, normalized by 20d return volatility; lag one day
rng=(high-low).replace(0,np.nan)
clv=((2*close-high-low)/rng).clip(-1,1)
raw=clv.rolling(3,min_periods=3).mean()
vol=close.pct_change().rolling(20,min_periods=15).std()
f=(raw/vol).shift(1)

def run(h):
    fr=close.shift(-h)/close-1
    vals=[]; dates=[]; ns=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
    a=np.array(vals); mean=a.mean(); sd=a.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    # date blocks
    def block(a,ds,st,en):
      q=a[(np.array(ds)>=pd.Timestamp(st))&(np.array(ds)<=pd.Timestamp(en))]
      return (len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252)) if len(q)>1 else (len(q),np.nan,np.nan)
    print('H',h,'IC %.6f ICIR %.6f dates %d avgN %.2f hit %.3f'% (mean,icir,len(a),np.mean(ns),np.mean(a>0)))
    for b in [('2020','2026-12-31'),('2027','2030-12-31'),('2031','2034-12-31'),('2035','2035-05-27')]: print(b,block(a,dates,b[0],b[1]))
    return mean,icir,dates,a
for h in [5,10,20]: run(h)
print('coverage',f.notna().sum(axis=1).mean()/15,'latest',f.loc[f.index[-1]].round(2).to_dict())
