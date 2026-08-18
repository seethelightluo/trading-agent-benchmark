import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
    px[s]=d
P=pd.DataFrame(px).sort_index()
r=P.pct_change()
# Lagged, volatility-normalized 10-session reversal; volatility uses only history through t-1.
ret10=P/P.shift(10)-1
vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20)
sig=(-(ret10/vol20)).shift(1)
# cross-sectional centering keeps only relative opportunity
sig=sig.sub(sig.mean(axis=1),axis=0)
future=P.shift(-10)/P-1
rows=[]; obs=[]
for dt in P.index:
    z=pd.concat([sig.loc[dt],future.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
        for s in z.index: obs.append((dt,s,float(sig.loc[dt,s])))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
rank=sig.rank(pct=True,axis=1)
print(json.dumps({'factor':'volnorm_short_reversal','horizon':10,'dates':len(q),'avg_n':round(q.n.mean(),2),'avg_valid_assets':round(q.n.mean(),2),'coverage':round(len(obs)/(len(P)*len(U)),4),'ic':round(q.ic.mean(),6),'icir':round(q.ic.mean()/q.ic.std()*np.sqrt(252),4),'hit':round((q.ic>0).mean(),4),'turnover':round(rank.diff().abs().mean().mean(),4)},indent=2))
for start in ['2028-01-01','2031-01-01','2032-01-01','2032-08-01']:
    z=q.loc[start:]
    if len(z): print(start,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std()*np.sqrt(252),4))
os.makedirs('scripts/artifacts',exist_ok=True)
pd.DataFrame(obs,columns=['date','symbol','signal']).to_csv('scripts/artifacts/miner_2_20330217_volnorm_short_reversal_signal.csv',index=False)
