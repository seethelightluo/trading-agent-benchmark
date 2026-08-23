import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: short-term reversal, amplified when cross-sectional 5d return dispersion is high.
# Factor at t uses closes through t-1; forward return starts t+1.
px={}
for s in U:
    try:
        d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
        px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
    except FileNotFoundError:
        pass
P=pd.DataFrame(px).sort_index().ffill()
r5=P.pct_change(5); r20=P.pct_change(20); vol20=P.pct_change().rolling(20).std()
disp=r5.std(axis=1, skipna=True)
# robust bounded dispersion gate relative to 60d median, avoids scale regime drift
med=disp.rolling(60,min_periods=30).median()
gate=(disp/(med+1e-12)).clip(0.5,2.0)
factor=(-r5*gate).shift(1)
fwd=P.shift(-10)/P-1
rows=[]
for dt in factor.index:
    a=factor.loc[dt]; b=fwd.loc[dt]
    z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
        rows.append((dt,ic,len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); res['date']=pd.to_datetime(res['date'])
def stats(x):
    return float(x.mean()), float(x.mean()/x.std(ddof=1)) if len(x)>1 and x.std(ddof=1)>0 else np.nan, float((x>0).mean()), len(x)
print('dates',len(res),'range',res.date.min().date(),res.date.max().date(),'avg_n',res.n.mean(),'coverage',res.n.mean()/15)
print('full IC/ICIR/hit/n',stats(res.ic))
for label,sub in [('365',res.tail(365)),('180',res.tail(180)),('90',res.tail(90)),('2028',res[res.date.dt.year==2028]),('2029',res[res.date.dt.year==2029]),('2030',res[res.date.dt.year==2030]),('2031',res[res.date.dt.year==2031]),('2032',res[res.date.dt.year==2032])]:
    print(label,stats(sub.ic))
# signal turnover as average rank top/bottom changes
rank=factor.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna()
print('turnover',turn.mean())
for h in [1,5,10,20]:
    fw=P.shift(-h)/P-1; vals=[]
    for dt in factor.index:
      z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
    print('horizon',h,'IC',np.nanmean(vals))
# artifacts for audit
out=factor.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20320205_dispersion_gated_short_reversal_signal.csv')
res.to_csv('scripts/miner_1_20320205_dispersion_gated_short_reversal_ic.csv',index=False)
