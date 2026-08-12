import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-11-04')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index().loc[:CUT]
PX={s:load(s) for s in U}; P=pd.concat(PX,axis=1).sort_index(); R=P.pct_change(); cs=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0); vol=R.rolling(60).std()*np.sqrt(20); F=(cs/(vol+.01)).shift(1)
rows=[]
for dt in F.index:
 z=pd.DataFrame({'f':F.loc[dt],'r':R.loc[R.index>dt].iloc[0] if (R.index>dt).any() else np.nan}).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.r).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('candidate=20d idiosyncratic momentum vs cross-asset median, vol scaled, lag1'); print('dates',len(d),'avgN',round(d.n.mean(),2),'IC',round(d.ic.mean(),6),'ICIR',round(d.ic.mean()/d.ic.std(ddof=1),6),'hit',round((d.ic>0).mean(),4),'coverage',round(d.n.sum()/(len(d)*15),4)); print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex(d.index).mean(),4))
for y,g in d.groupby(d.index.year): print(y,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
print('last usable',P.index.max().date())
