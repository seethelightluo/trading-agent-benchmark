import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: x=None
        if x is not None and len(x): break
    if x is not None and len(x):
        x=x.copy(); x.date=pd.to_datetime(x.date)
        D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lp=np.log(p); r=lp.diff()
# Risk-adjusted medium momentum: trailing 20-session return divided by 20-session realized volatility,
# with all inputs lagged one completed session.
ret20=lp-lp.shift(20); vol20=r.rolling(20,min_periods=12).std()*np.sqrt(20)
f=(ret20/vol20).shift(1)
fr10=lp.shift(-10)-lp; fr1=lp.shift(-1)-lp
rows=[]
for dt in f.index:
    a=f.loc[dt]
    for h,b in [(10,fr10.loc[dt]),(1,fr1.loc[dt])]:
        ok=a.notna()&b.notna()
        if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:
            rows.append((dt,h,a[ok].corr(b[ok]),int(ok.sum())))
z=pd.DataFrame(rows,columns=['date','horizon','ic','n'])
print('shape',p.shape,'assets',len(D),'dates',p.index.min(),p.index.max())
for h in [1,10]:
 q=z[z.horizon==h].set_index('date').ic
 nn=z[z.horizon==h].set_index('date').n
 print('H%d valid_dates %d avgN %.2f coverage %.4f IC %.8f ICIR %.8f hit %.4f'%(h,len(q),nn.mean(),nn.mean()/len(D),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
  x=q.loc[lo:hi]; print(' regime',lo,len(x),'IC %.8f ICIR %.8f'%(x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan))
 print(' recent120',q.tail(120).mean(),q.tail(120).mean()/q.tail(120).std(ddof=1))
 if h==10:
  f.to_csv('scripts/miner_2_20311225_volnorm_momentum20_signal.csv')
  z[z.horizon==10].to_csv('scripts/miner_2_20311225_volnorm_momentum20_ic.csv',index=False)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
