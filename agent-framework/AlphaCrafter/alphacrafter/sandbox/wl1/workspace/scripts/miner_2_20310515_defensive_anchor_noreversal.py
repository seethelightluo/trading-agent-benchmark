import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000300.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];U=list(dict.fromkeys(U));P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();v=r.rolling(20,min_periods=15).std();m=r.rolling(20,min_periods=15).sum(); med=m.median(axis=1)
# Defensive anchor changes the cross-sectional momentum exposure according to gold and
# long-duration yield relative strength; no short-term reversal overlay.
a=((m['XAU']-med)+(m['US10Y']-med)+(m['CN10Y']-med))/3
st=np.tanh(a/(v.median(axis=1)+1e-6)); rel=m.sub(med,axis=0).div(v+1e-6); f=rel.mul(1-.30*st,axis=0)+.20*st*rel
f=f.sub(f.median(axis=1),axis=0).shift(1); rows=[]
for i in range(len(px)-20):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+10]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:rows.append((px.index[i],len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','n','ic']);q=o.set_index('date').ic
print('period',px.index.min(),px.index.max(),'assets',px.shape[1],'dates',len(q),'avgN',o.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
 if len(sub)>20:print(label,len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()));f.to_csv('scripts/miner_2_20310515_defensive_anchor_noreversal_signal.csv')
