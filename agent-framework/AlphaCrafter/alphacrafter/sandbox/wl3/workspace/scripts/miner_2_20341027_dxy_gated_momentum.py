import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s, n=4200):
    d=(get_index_daily_data(s,n) if s=='DXY' else get_stock_daily_data(s,n))
    if d is None or len(d)==0: return pd.Series(dtype=float)
    return pd.Series(d['close'].values,index=pd.to_datetime(d['date']))
px={s:load(s) for s in assets}; dxy=load('DXY')
# signal: medium-term momentum, activated and signed by DXY trend (risk-on when DXY falling)
common=sorted(set.intersection(*[set(x.index) for x in px.values() if len(x)]) & set(dxy.index))
P=pd.DataFrame({s:px[s].reindex(common) for s in assets}); D=dxy.reindex(common)
rets=np.log(P).diff(); dr=np.log(D).diff()
# DXY 20d trend, negative means supportive risk appetite; signal asset momentum with macro tilt
mom=np.log(P/P.shift(60)); macro=-np.sign(dr.rolling(20).sum())
# continuous macro gate: momentum is trusted when DXY trend is weak/falling, attenuated when rising
z=dr.rolling(20).sum(); gate=np.tanh(-z/(dr.rolling(60).std()*np.sqrt(20)+1e-12))
factor=mom.mul(gate,axis=0)
# cross-sectional demean, forward 10d return
fwd=np.log(P.shift(-10)/P)
ics=[]; dates=[]
for t in factor.index:
    a=factor.loc[t]; b=fwd.loc[t]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        ics.append(a[ok].corr(b[ok])); dates.append(t)
ic=pd.Series(ics,index=dates).dropna()
# turnover rank changes, coverage
rank=factor.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(ic),'avg_n', round(np.mean([((factor.loc[t].notna())&(fwd.loc[t].notna())).sum() for t in dates]),2))
print('coverage',round(factor.notna().mean().mean(),4),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'turnover',round(turnover,4))
for w in [120,252,504]:
 q=ic.tail(w); print('recent',w,'n',len(q),'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
for h in [1,5,10,20]:
 ff=np.log(P.shift(-h)/P); ii=[]
 for t in factor.index:
  ok=factor.loc[t].notna()&ff.loc[t].notna()
  if ok.sum()>=8: ii.append(factor.loc[t,ok].corr(ff.loc[t,ok]))
 q=pd.Series(ii).dropna(); print('decay',h,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
print('period',common[0],common[-1])
