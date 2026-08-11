import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct=get_account_dict(); uni=acct.get('watch_list',[])
if not uni: uni=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in uni:
    x=get_stock_daily_data(s,days=2200)
    if x is None or len(x)<80: x=get_index_daily_data(s,days=2200)
    if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill()
r=px.pct_change()
# factor: medium-term return divided by downside deviation, with a mild recovery penalty
ret20=px/px.shift(20)-1
neg=r.where(r<0,0).rolling(20).apply(lambda z: np.sqrt(np.mean(z*z)),raw=True)
f=ret20/(neg+1e-5)
# winsorize cross-section, lag one completed day implicit via t -> t+1
rows=[]
for i in range(len(px)-1):
    vals=f.iloc[i]; fw=r.iloc[i+1]
    z=pd.concat([vals.rename('f'),fw.rename('y')],axis=1).dropna()
    if len(z)>=8:
        z['f']=z.f.clip(z.f.quantile(.05),z.f.quantile(.95))
        rows.append((px.index[i],len(z),z.f.corr(z.y),z.f.corr(r.iloc[i+1:i+4].sum()),z.f.corr(r.iloc[i+1:i+6].sum()),z.f.corr(r.iloc[i+1:i+11].sum())))
a=pd.DataFrame(rows,columns=['date','n','ic1','ic3','ic5','ic10']).set_index('date')
for col in ['ic1','ic3','ic5','ic10']:
    x=a[col].dropna(); print(col,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit', (x>0).mean())
# turnover rank signal, and regime split based equal-weight prior 20d
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean(),'coverage',f.notna().mean().mean(),'period',a.index.min(),a.index.max())
# annual/regime blocks
for name,mask in [('2020-22',(a.index<'2023-01-01')),('2023-25',((a.index>='2023-01-01')&(a.index<'2026-01-01'))),('2026-27',(a.index>='2026-01-01'))]:
 x=a.loc[mask,'ic1'].dropna(); print(name,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
# macro state: market trend
m=r.mean(axis=1).rolling(20).sum(); q=a.join(m.rename('m'),how='left');
for name,mask in [('up',q.m>0),('down',q.m<=0)]:
 x=q.loc[mask,'ic1'].dropna(); print(name,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
