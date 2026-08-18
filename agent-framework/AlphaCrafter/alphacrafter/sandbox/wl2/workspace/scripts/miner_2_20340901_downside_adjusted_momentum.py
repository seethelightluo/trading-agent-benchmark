import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    p=f'../persistent/stock_data/{s}.csv'
    if os.path.exists(p):
        x=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
        D[s]=x[x.index<=pd.Timestamp('2034-09-01')]
prices=pd.DataFrame(D).sort_index()
# single interpretable idea: lagged 20d return divided by downside volatility (losses only), with 60d shrinkage
ret=prices.pct_change()
r20=prices.pct_change(20)
down=ret.where(ret<0,0).rolling(40,min_periods=20).std()*np.sqrt(252)
signal=(r20/down.replace(0,np.nan)).shift(1)
# forward horizons from close t to close t+h
out=[]
for h in [1,5,10,20,40]:
    fwd=prices.shift(-h)/prices-1
    ics=[]; nobs=[]; cov=[]
    for dt in signal.index:
        a=pd.concat([signal.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            ics.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); nobs.append(len(a)); cov.append(len(a)/15)
    z=pd.Series(ics).dropna(); ic=z.mean(); sd=z.std(ddof=1)
    print(f'h={h} dates={len(z)} avg_n={np.mean(nobs):.2f} coverage={np.mean(cov):.4f} IC={ic:.6f} ICIR={ic/sd*np.sqrt(252):.6f} paperICIR={ic/sd:.6f} hit={np.mean(z>0):.4f}')
# recent regime and turnover for selected horizon
h=10; fwd=prices.shift(-h)/prices-1
for start,end in [('2020-01-01','2026-07-15'),('2026-07-16','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-08-31')]:
 z=[]
 for dt in signal.loc[start:end].index:
  a=pd.concat([signal.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('regime',start,end,'dates',len(z),'IC',z.mean(),'paperICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
rank=signal.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('overall turnover proxy',turn,'valid assets',signal.notna().mean().mean(),'last',signal.iloc[-1].dropna().to_dict())
# save reproducible signal artifact
signal.to_csv('../persistent/miner_2_20340901_downside_adjusted_momentum_10d_signal.csv')
