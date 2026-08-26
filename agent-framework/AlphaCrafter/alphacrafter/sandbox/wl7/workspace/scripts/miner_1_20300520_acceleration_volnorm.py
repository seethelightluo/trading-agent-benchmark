import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
acct=get_account_dict(); wl=acct.get('watch_list') or U
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=4000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=x.close.astype(float)
px=pd.DataFrame(frames).sort_index().ffill()
# factor: lagged volatility-normalized acceleration, short return minus slower return
# computed with information through t-1; cross-sectional prediction of t -> t+10
r5=px.pct_change(5); r20=px.pct_change(20); vol=px.pct_change().rolling(20).std()
factor=((r5-r20)/vol).shift(1)
fwd=px.shift(-10)/px-1
rows=[]; dates=[]
for dt in px.index:
    a=factor.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
# trailing valid dates, and regimes
for label, rr in [('full',res),('early',res.iloc[:len(res)//3]),('mid',res.iloc[len(res)//3:2*len(res)//3]),('late',res.iloc[2*len(res)//3:])]:
    mean=rr.ic.mean(); sd=rr.ic.std(ddof=1); print(label,'dates',len(rr),'avg_n',rr.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252) if sd else np.nan,'hit',(rr.ic>0).mean())
# horizon decay
for h in [1,5,10,20,40]:
    fw=px.shift(-h)/px-1; vals=[]
    for dt in px.index:
      z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print('decay',h,'IC',np.nanmean(vals),'dates',len(vals))
# rank turnover and coverage
rank=factor.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
print('cutoff',px.index[-1], 'assets',len(frames),'dates',len(px),'coverage',factor.notna().sum().sum()/(factor.shape[0]*len(U)),'turnover',turnover)
res.to_csv('scripts/miner_1_20300520_acceleration_volnorm_ic.csv')
# signal artifact: factor observations for audit
factor.to_csv('scripts/miner_1_20300520_acceleration_volnorm_signal.csv')
