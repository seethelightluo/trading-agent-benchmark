import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    d=get_stock_daily_data(s, days=1800)
    if d is None or len(d)<300: d=get_index_daily_data(s, days=1800)
    return d
raw={s:get(s) for s in U}; raw={s:d for s,d in raw.items() if d is not None}
# align by date, factor: 15d average close-location (close-low)/(high-low), centered to [-1,1], times 20d return, lagged
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in raw.items()})
clv={}
for s,d in raw.items():
    x=d.set_index('date'); rg=(x.high-x.low).replace(0,np.nan)
    clv[s]=((x.close-x.low)/rg*2-1).rolling(15,min_periods=10).mean()
clv=pd.DataFrame(clv).reindex(px.index)
ret=px.pct_change(20); vol=px.pct_change().rolling(20).std()
f=(clv*ret/vol.replace(0,np.nan)).shift(1)
# forward returns
rows=[]
for h in [1,5,10,20]:
    fw=px.pct_change(h).shift(-h)
    vals=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); dates.append(dt)
    a=np.array(vals); print('h',h,'IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),len(a),np.nanmean([len(pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()) for d in dates])))
# turnover rank signal, coverage
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('assets',len(raw),'date range',px.index.min(),px.index.max(),'rows',len(px),'turnover',turn,'coverage',f.notna().sum(axis=1).mean()/len(U))
for yr in [2021,2022,2023,2024,2025,2026,2027,2028,2029]:
 a=[]; fw=px.pct_change().shift(-1)
 for dt in f.index[f.index.year==yr]:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 if a: print('yr',yr,'ic %.5f icir %.4f n %d'%(np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),len(a)))
