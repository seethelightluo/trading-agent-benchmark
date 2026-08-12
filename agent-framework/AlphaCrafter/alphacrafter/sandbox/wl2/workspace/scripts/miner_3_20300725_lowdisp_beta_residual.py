import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
# Beta-residual 5-day reversal, normalized by idiosyncratic volatility; only completed t predicts t+1.
vol=r.rolling(30,min_periods=15).std(); cov=r.rolling(60,min_periods=30).cov(m); vm=m.rolling(60,min_periods=30).var()
beta=cov.div(vm,axis=0); resid=r-beta.mul(m,axis=0)
f=-resid.rolling(5,min_periods=5).sum()/vol
# condition on cross-sectional dispersion being below its trailing median (distinct from high-dispersion factor)
disp=r.std(axis=1); gate=(disp<disp.rolling(120,min_periods=60).median()).astype(float)
f=f.mul(gate,axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  z.f=z.f.clip(z.f.quantile(.05),z.f.quantile(.95)); rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=a.ic.dropna()
print('dates',len(x),'avgN',round(a.n.mean(),3),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',a.index.min(),a.index.max())
for name,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 q=a.loc[mask].ic.dropna(); print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1); rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.clip(z.f.quantile(.05),z.f.quantile(.95)).corr(z.y))
 print('h',h,'IC',np.nanmean(rr),'n',len(rr))
# Save signal artifact for provenance
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20300725_lowdisp_beta_residual_signal.csv')
