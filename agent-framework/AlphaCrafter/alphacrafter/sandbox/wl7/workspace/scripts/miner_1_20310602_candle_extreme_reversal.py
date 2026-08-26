import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80:
  z=d[['date','open','high','low','close']].copy(); z.date=pd.to_datetime(z.date); raw[s]=z.set_index('date')
P=pd.DataFrame({s:z.close for s,z in raw.items()}).sort_index(); H=pd.DataFrame({s:z.high for s,z in raw.items()}).reindex(P.index); L=pd.DataFrame({s:z.low for s,z in raw.items()}).reindex(P.index)
r=P.pct_change(); v=r.rolling(20,min_periods=10).std().shift(1)
# One-day contrarian return, strengthened when the completed candle closes at an extreme.
loc=((P-L)/(H-L).replace(0,np.nan)).clip(0,1)
sig=(-(r/(v+1e-12))*(1+2*(loc-.5).abs())).shift(1)
sig=sig.sub(sig.median(axis=1),axis=0)
def test(h):
 y=P.shift(-h)/P-1; vals=[]; rows=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'));rows.append((dt,vals[-1],int(ok.sum())))
 a=pd.Series(vals);return a,rows
for h in [1,5,10,20]:
 a,rr=test(h);print('h',h,'dates',len(a),'avg_n %.2f'%np.mean([x[2] for x in rr]),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,rr=test(1);print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean()));print('regimes',[a.iloc[i:j].mean() for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]])
out='scripts/miner_1_20310602_candle_extreme_reversal';pd.DataFrame(rr,columns=['date','ic','n']).to_csv(out+'_ic.csv',index=False);sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv(out+'_signal.csv',index=False);print('signal_artifact='+out+'_signal.csv')
