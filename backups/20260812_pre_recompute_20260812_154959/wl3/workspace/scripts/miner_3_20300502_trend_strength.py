import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    x=get_stock_daily_data(s, days=4000)
    if x is None or len(x)==0: x=get_index_daily_data(s, days=4000)
    return x[['date','close']].drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.concat({s:get(s) for s in U},axis=1).sort_index().ffill()
r=np.log(px).diff()
# trend-strength adjusted residual momentum: 20d return relative to cross-sectional median,
# scaled by 60d vol and persistence of positive daily returns, all lagged one day.
csmed=r.rolling(20).sum().median(axis=1)
raw=r.rolling(20).sum().sub(csmed,axis=0)
vol=r.rolling(60).std().replace(0,np.nan)
persist=(r.gt(0).rolling(40).mean()-0.5)*2
f=(raw/vol*persist).shift(1)
# forward non-overlapping daily forward returns at horizons
for h in [1,3,5,10]:
    fr=np.log(px.shift(-h)/px)
    vals=[]; dates=[]; ns=[]
    for d in f.index:
        z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); ns.append(len(z))
    a=np.asarray(vals); print('H',h,'obs',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
    # recent and regimes
    for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-02')]:
        q=a[(np.array(dates)>=pd.Timestamp(lo)) & (np.array(dates)<=pd.Timestamp(hi))]
        if len(q): print(' ',lo,hi,'n',len(q),'ic',round(q.mean(),5),'ir',round(q.mean()/(q.std(ddof=1)+1e-12),5))
print('coverage',round(f.notna().mean().mean(),4),'dates',len(f))
# signal artifact for primary 5d
out=f.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_3_20300502_trend_strength_signal.csv',index=False)
