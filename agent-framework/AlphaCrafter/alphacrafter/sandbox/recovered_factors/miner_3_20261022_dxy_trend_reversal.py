import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(dxy.index))
wide=pd.DataFrame({a:px[a].reindex(dates) for a in assets}); m=dxy.reindex(dates)
r1=wide.pct_change(1); dr5=m.pct_change(5)
# DXY-trend-conditioned short-term reversal: reversal is amplified during dollar moves
mult=(1+10*dr5.clip(-.03,.03)).clip(.7,1.3)
f=-r1.mul(mult,axis=0)
fwd=wide.shift(-1).div(wide)-1
# libraries
libs=[]
for fn in glob.glob('factors/*.json'):
 try:
  q=json.load(open(fn)); libs.append(q['factor_id'])
 except: pass
# approximate admitted library signals from definitions
sig={}
sig['miner_3_risk_adjusted_trend_20d']=wide.pct_change(20).div(wide.pct_change().rolling(20).std())
sig['miner_3_relative_volume_participation_20d']=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['volume'].reindex(dates).rolling(20).mean().values for a in assets})
sig['miner_1_ravmom_20obs']=wide.pct_change(20).div(wide.pct_change().rolling(20).std())
sig['miner_1_volnorm_reversal_5obs']=-wide.pct_change(5).div(wide.pct_change().rolling(20).std())
sig['miner_2_realized_volatility_20obs']=-wide.pct_change().rolling(20).std()
sig['miner_3_vix_conditioned_reversal_1d']=-r1.mul(pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(dates).div(pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(dates).rolling(60).median()).clip(0.2,5),axis=0)
def ics(x,y,h=1):
 z=[]
 for dt in dates:
  a=x.loc[dt]; b=y.shift(-h).loc[dt] if dt in y.index else None
  q=pd.concat([a,b],axis=1).dropna()
  if len(q)>=8:z.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
 return pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
for h in [1,5,10]:
 z=ics(f,fwd,h); print('H',h,'dates',len(z),'meanIC %.6f ICIR %.6f hit %.3f meanN %.2f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean(),z.n.mean()))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
mx=0; detail={}
for k,v in sig.items():
 q=pd.concat([f.stack(),v.reindex(index=dates,columns=assets).stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic; detail[k]=rho;mx=max(mx,abs(rho))
print('maxcorr',mx,detail)
for y in [2020,2021,2022,2023,2024,2025,2026]:
 z=ics(f,fwd,1); z=z[z.index.year==y]; print(y,len(z),round(z.ic.mean(),5))
print('period',dates[0],dates[-1])
