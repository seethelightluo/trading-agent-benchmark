import numpy as np, pandas as pd
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
        x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Observation-only VIX/DXY macro regime: fade cross-sectional shocks only when
# macro stress is rising, while orthogonalizing asset returns to the cross-section.
def macro(sym):
    x=pd.read_csv('../persistent/index_data/'+sym+'.csv')
    x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date').set_index('date')
    col='close' if 'close' in x else x.select_dtypes('number').columns[0]
    return pd.to_numeric(x[col],errors='coerce').reindex(p.index).ffill()
vix=macro('VIX'); dxy=macro('DXY')
# stress impulse is lagged and robustly standardized against 60d history
vixret=vix.pct_change(3); dxyret=dxy.pct_change(3)
stress=(vixret>0)&(dxyret>0)
res=r.sub(r.median(axis=1),axis=0)
f=-res/(r.rolling(20).std()+1e-12)*stress.astype(float).values[:,None]
for h in [1,3,5,6,10]:
 rows=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rows.append((p.index[i],z.f.corr(z.y),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),3),'coverage',round(a.n.sum()/(len(a)*15),4),'IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
  q=a.loc[lo:hi]
  if len(q): print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True); print('turnover',((rank.diff().abs().mean(axis=1)/2).mean()),'signal_coverage',f.notna().mean().mean(),'last_date',p.index[-1].date())
f.index.name='date'; f.reset_index().to_csv('scripts/miner_1_20310612_macro_stress_reversal_signal.csv',index=False)
