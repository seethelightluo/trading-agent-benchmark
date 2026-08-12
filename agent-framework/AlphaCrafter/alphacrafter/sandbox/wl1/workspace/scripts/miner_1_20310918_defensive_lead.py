import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
RISK=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','COPPER','WTI','BTC','ETH']

def fetch(s):
    
    try: d=get_stock_daily_data(s, days=5000)
    except Exception: d=get_index_daily_data(s, days=5000)
    if d is None or len(d)==0:return pd.Series(dtype=float)
    x=d.copy(); x['date']=pd.to_datetime(x['date']); return x.set_index('date')['close'].astype(float).sort_index()
px=pd.DataFrame({s:fetch(s) for s in ASSETS}).sort_index()
# macro observation-only VIX, used only as conditioning signal

try: v=get_index_daily_data('VIX', days=5000)
except Exception: v=None
if v is None: vix=pd.Series(index=px.index,dtype=float)
else:
    v=v.copy(); v['date']=pd.to_datetime(v['date']); vix=v.set_index('date')['close'].astype(float).sort_index().reindex(px.index).ffill(); vix.name='VIX'
ret=np.log(px).diff()
mom20=np.log(px/px.shift(20)); riskret=mom20[RISK].mean(axis=1)
rel=mom20.sub(riskret,axis=0)
vol20=ret.rolling(20,min_periods=10).std()*np.sqrt(20)
# stress intensity from trailing VIX percentile, no lookahead; shift signal later
vp=vix.rolling(252,min_periods=100).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1])
stress=((vp-0.55)/0.35).clip(0,1)
# defensive relative leadership: normal trend, stress relative leadership, volatility scaled
raw=(1-stress).fillna(0)*mom20 + stress.fillna(0)*rel
fac=raw/(vol20+1e-8)
fac=fac.shift(1)
# forward returns aligned
out=[]
for h in [1,5,10,20]:
 fwd=np.log(px.shift(-h)/px)
 vals=[]
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 ic=z.ic
 out.append((h,len(z),z.n.mean(),ic.mean(),ic.std(ddof=1),ic.mean()/ic.std(ddof=1), (ic>0).mean()))
print('dates',len(px),'assets',px.notna().sum().describe().to_dict())
print('RESULTS h obs avgN IC ICIR hit')
for x in out: print('%d %d %.2f %.6f %.6f %.6f %.4f'%x)
# regimes by calendar
fwd=np.log(px.shift(-20)/px)
vals=[]
for dt in fac.index:
 a=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(a)>=8: vals.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
for label,sub in [('2020-22',z.loc['2020':'2022']),('2023-25',z.loc['2023':'2025']),('2026-28',z.loc['2026':'2028']),('2029-30',z.loc['2029':'2030']),('2031YTD',z.loc['2031':])]:
 print(label,len(sub),sub.n.mean(),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
# coverage and rank turnover proxy
coverage=fac.notna().sum(axis=1).mean()/len(ASSETS)
ranks=fac.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).mean()
print('coverage',coverage,'turnover_proxy',turn)
# artifact signals for audit, last available values
fac.to_csv('scripts/miner_1_20310918_defensive_lead_signal.csv',index_label='date')
