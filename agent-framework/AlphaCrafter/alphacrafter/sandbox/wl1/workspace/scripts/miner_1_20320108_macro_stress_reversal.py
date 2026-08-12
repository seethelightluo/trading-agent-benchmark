import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=1500) for s in U}; px={s:d.set_index('date')['close'].astype(float) for s,d in px.items() if d is not None}
c=pd.concat(px,axis=1).sort_index().ffill(); r=np.log(c).diff(); v=get_index_daily_data('VIX',days=1500).set_index('date')['close'].astype(float).reindex(c.index).ffill()
res=r.rolling(20).sum(); res=res.sub(res.median(axis=1),axis=0); vol=r.rolling(20).std()*np.sqrt(20); vz=(v-v.rolling(120).mean())/(v.rolling(120).std()+1e-12); f=(-res/(vol+1e-12)).mul((1+.35*np.tanh(vz)).values,axis=0)
def calc(h):
 y=np.log(c.shift(-h)/c); vals=[]
 for i in range(len(f)-h):
  x=f.iloc[i].values; z=y.iloc[i].values; ok=np.isfinite(x)&np.isfinite(z)
  if ok.sum()>=8: vals.append(np.corrcoef(x[ok],z[ok])[0,1])
 return np.array(vals)
ic=calc(20); ix=f.index[20:20+len(ic)]
print('assets',len(px),'dates',len(ic),'avg_n',len(px),'coverage',len(ic)/(len(f)-20),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for h in [1,5,10,20,40]: print('decay',h,calc(h).mean())
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 z=pd.Series(ic,index=ix).loc[a:b]; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
f.index.name='date';f.reset_index().to_csv('scripts/miner_1_20320108_macro_stress_reversal_signal.csv',index=False)
