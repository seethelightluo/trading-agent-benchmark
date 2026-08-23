import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

UNIVERSE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in UNIVERSE:
    f=os.path.join(base,s+'.csv')
    d=pd.read_csv(f)
    d['date']=pd.to_datetime(d['date'])
    px[s]=d.set_index('date')['close'].sort_index()
prices=pd.DataFrame(px).sort_index()
r=prices.pct_change()
# Candidate: fade unusually large recent 5-day move, with risk scaling by trailing 20d volatility.
# Higher score means expected 10d return.
vol=r.rolling(20).std()
factor=-(prices/prices.shift(5)-1)/vol
fwd=prices.shift(-10)/prices-1
rows=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
res=res[(res.index>='2020-01-01') & (res.index<='2030-10-02')]
mu=res.ic.mean(); sd=res.ic.std(ddof=1)
print(json.dumps({'factor':'volatility_shock_reversal','valid_dates':len(res),'average_instruments':res.n.mean(),'coverage':res.n.mean()/15,'ic':mu,'icir':mu/sd*np.sqrt(252) if sd else None,'hit':(res.ic>0).mean(),'turnover_proxy':None,'regimes':{str(y):res.loc[res.index.year==y,'ic'].mean() for y in sorted(res.index.year.unique())},'decay':{}},default=str))
