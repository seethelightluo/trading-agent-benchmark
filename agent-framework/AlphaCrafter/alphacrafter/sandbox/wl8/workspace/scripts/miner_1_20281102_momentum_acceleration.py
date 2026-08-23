import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-10-31')
P={}
for s in U:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Volatility-normalized momentum acceleration: recent 5d return minus the
# average daily return over the preceding 20d window, lagged one day.
recent=r.rolling(5,min_periods=5).sum()
prior=r.shift(5).rolling(20,min_periods=15).sum()/4.0
vol=r.rolling(20,min_periods=15).std()
sig=((recent-prior)/vol).shift(1).clip(-8,8)
print('panel_dates',len(px),'assets',px.shape[1],'end',px.index.max().date())
for h in [1,3,5,10]:
    f=px.shift(-h)/px-1; rows=[]
    for d in px.index:
        g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
        if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
            rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
    z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.tail(180)
    print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20281102_momentum_acceleration_signal.csv',index=False)
