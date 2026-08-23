import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,3000)
    except Exception: pass
    if d is None or len(d)<300:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is not None and len(d)>300:
        x=d.copy(); x.date=pd.to_datetime(x.date)
        px[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=np.log(P).diff()
# 20-day lagged momentum scaled by trailing realized volatility; signal uses data through t-1.
ret20=P.shift(1).pct_change(20)
vol20=r.shift(1).rolling(20).std()*np.sqrt(252)
f=ret20/(vol20+1e-12)
fwd=P.shift(-10)/P-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        rows.append((dt,a[ok].rank().corr(b[ok].rank()),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
for name,x in [('full',z),('recent365',z.tail(365)),('recent180',z.tail(180)),('recent60',z.tail(60)),('2029',z.loc['2029']),('2030',z.loc['2030']),('2031',z.loc['2031'])]:
    if len(x): print(name,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/(x.ic.std(ddof=1)+1e-12)*np.sqrt(252),6),'hit',round((x.ic>0).mean(),4))
q=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(q)):
    a=q.iloc[i]; b=q.iloc[i-1]; ok=a.notna()&b.notna()
    if ok.sum()>=8: turn.append((a[ok]-b[ok]).abs().mean())
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(np.mean(turn),6),'instruments',len(px),'dates',len(z),'last',z.index.max().date())
f.to_csv('scripts/miner_1_20310724_volatility_scaled_momentum_signal.csv')
pd.DataFrame({'date':z.index,'ic':z.ic,'n':z.n}).to_csv('scripts/miner_1_20310724_volatility_scaled_momentum_ic.csv',index=False)
