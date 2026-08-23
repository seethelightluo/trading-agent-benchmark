import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

root=Path('../persistent/stock_data')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
    d=pd.read_csv(root/f'{s}.csv')
    d['date']=pd.to_datetime(d['date'])
    px[s]=d.set_index('date')['close'].astype(float)
prices=pd.concat(px,axis=1).sort_index()
prices=prices.loc[:'2032-09-15']
rets=prices.pct_change()
# Candidate: medium-short momentum per realized risk, with mild trend persistence
signal=prices.pct_change(10).div(rets.rolling(30).std().replace(0,np.nan))
# forward 10 trading-day return
fwd=prices.shift(-10).div(prices)-1
rows=[]
for dt in signal.index:
    x=signal.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=10d return / 30d realized vol; horizon=10d')
print('dates',len(r),'avg_n',r.n.mean(),'cell_coverage',signal.notna().sum().sum()/np.prod(signal.shape))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), signal.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10,20,40]:
    fw=prices.shift(-h).div(prices)-1; vals=[]
    for dt in signal.index:
      z=pd.concat([signal.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    a=np.array(vals); print('decay',h,'n',len(a),'IC %.6f ICIR %.6f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)))
r2=r.copy(); r2['year']=r2.index.year
print('year_IC'); print(r2.groupby('year').ic.mean().to_string())
