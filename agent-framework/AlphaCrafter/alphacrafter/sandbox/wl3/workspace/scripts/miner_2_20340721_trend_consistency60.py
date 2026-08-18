import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    p=os.path.join(base,s+'.csv')
    if os.path.exists(p):
        d=pd.read_csv(p)
        d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# signal: trend persistence: 20d return, multiplied by directional consistency over 60d,
# with volatility normalization; all shifted one completed day
r=P.pct_change()
ret20=P/P.shift(20)-1
cons=(r.rolling(60,min_periods=45).mean()/r.rolling(60,min_periods=45).std()).clip(-3,3)
# interpretable sign-persistence (net positive-day fraction), preserving trend direction
signpersist=r.gt(0).rolling(60,min_periods=45).mean()*2-1
vol=r.rolling(30,min_periods=20).std()
f=(ret20*signpersist/vol).shift(1)
rows=[]
for h in [1,3,5,10,20]:
    fr=P.shift(-h)/P-1
    vals=[]; dates=[]; ns=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
    a=np.array(vals); ic=a.mean(); ir=ic/a.std(ddof=1) if len(a)>1 else np.nan
    print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(a>0).mean():.4f}')
    if h==10:
      for n in [120,252,756,1260]:
       q=a[-n:]; print(f'recent{n}_ICIR={q.mean()/q.std(ddof=1):.8f} IC={q.mean():.8f} n={len(q)}')
# coverage and turnover on valid cross sections
valid=f.notna().sum(axis=1); print('dates',len(P),'coverage',valid.sum()/(len(P)*len(U)),'avgN',valid.mean())
# turnover rank signal: fraction changed among consecutive dates
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
# artifact full signal, long format
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340721_trend_consistency60_signal.csv',index=False)
