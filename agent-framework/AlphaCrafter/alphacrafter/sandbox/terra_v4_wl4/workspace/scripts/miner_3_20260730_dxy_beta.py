import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2026-07-15')
def load(sym):
    return pd.read_csv(f'../persistent/stock_data/{sym}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index().rename('DXY')
px=pd.concat([load(s).rename(s) for s in U]+[dxy],axis=1,join='inner').sort_index()
r=px.pct_change()
beta=r[U].rolling(60,min_periods=45).cov(r.DXY).div(r.DXY.rolling(60,min_periods=45).var(),axis=0)
f=-beta
# Forward returns are constructed within each asset's own observed trading sequence, avoiding calendar-row shifts.
def forward(h): return pd.concat([(px[s].shift(-h)/px[s]-1).rename(s) for s in U],axis=1)
def evalh(h):
    fw=forward(h); rows=[]
    for dt in f.index[f.index<=CUTOFF]:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
    return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
a=evalh(1)
print('range',a.index.min(),a.index.max(),'dates',len(a),'avg_n %.2f coverage %.4f'%(a.n.mean(),a.n.sum()/(len(a)*15)))
print('daily IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean()))
for h in [5,10]:
 q=evalh(h);print('%dd IC %.6f ICIR %.6f dates %d'%(h,q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
for y in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[(a.index.year>=y[0])&(a.index.year<=y[1])].ic;print(y,'mean %.6f icir %.6f n %d'%(q.mean(),q.mean()/q.std(),len(q)))
rank=f.loc[:CUTOFF].rank(axis=1,pct=True); print('turnover %.4f'%rank.diff().abs().mean().mean())
q=a.tail(250);print('recent250 IC %.6f ICIR %.6f n %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
f.loc[:CUTOFF].to_csv('scripts/miner_3_20260730_dxy_beta_signal.csv')
