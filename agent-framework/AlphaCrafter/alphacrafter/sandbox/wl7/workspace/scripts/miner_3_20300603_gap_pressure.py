import os, sys
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

out='scripts/miner_3_20300603_gap_pressure.py'
universe=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in universe:
    d=get_stock_daily_data(s,days=3000)
    if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
    if d is not None:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date'); frames[s]=d
print('assets',len(frames), 'lengths', {k:len(v) for k,v in frames.items()})
# Overnight gap pressure: signed 10d gap accumulation, normalized by 20d close volatility.
# Positive gaps that persist should indicate demand; lag one completed session.
sigs=[]
for s,d in frames.items():
    c=pd.to_numeric(d.close,errors='coerce'); o=pd.to_numeric(d.open,errors='coerce')
    gap=np.log(o/c.shift(1)).replace([np.inf,-np.inf],np.nan)
    lr=np.log(c/c.shift(1)); vol=lr.rolling(20).std()
    # separate gap from intraday close-to-open move; mild winsorization
    x=gap.clip(-.15,.15).rolling(10).sum()/(vol*np.sqrt(10))
    # signal as of t, prediction starts t+1
    sigs.append(pd.DataFrame({'date':d.date,'asset':s,'signal':x.shift(1),'close':c}))
all=pd.concat(sigs).sort_values(['date','asset'])
# forward 10 trading observations per asset
all['fwd10']=all.groupby('asset').close.shift(-10)/all.close-1
obs=[]
for dt,g in all.groupby('date'):
    z=g.dropna(subset=['signal','fwd10'])
    if len(z)>=8: obs.append((dt,len(z),z.signal.corr(z.fwd10,method='spearman')))
ic=pd.DataFrame(obs,columns=['date','n','ic']).dropna()
mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
# rank turnover on consecutive common dates
r=all.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
to=(r.diff().abs().mean(axis=1)/2).dropna().mean()
coverage=all.signal.notna().groupby(all.date).mean().mean()
print('dates',len(ic),'avg_n',ic.n.mean(),'IC %.8f ICIR %.8f hit %.4f turnover %.6f coverage %.4f'%(mean,icir,(ic.ic>0).mean(),to,coverage))
for h in [1,5,10,20,40]:
 vals=[]
 all['fh']=all.groupby('asset').close.shift(-h)/all.close-1
 for dt,g in all.groupby('date'):
  z=g.dropna(subset=['signal','fh'])
  if len(z)>=8: vals.append(z.signal.corr(z.fh,method='spearman'))
 vals=pd.Series(vals).dropna(); print('decay',h, 'IC %.8f ICIR %.8f'%(vals.mean(),vals.mean()/vals.std(ddof=1)*np.sqrt(252)))
for label,sub in [('early',ic.iloc[:len(ic)//3]),('mid',ic.iloc[len(ic)//3:2*len(ic)//3]),('late',ic.iloc[2*len(ic)//3:])]: print(label,len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252))
all.to_csv('scripts/miner_3_20300603_gap_pressure_signal.csv',index=False); ic.to_csv('scripts/miner_3_20300603_gap_pressure_ic.csv',index=False)
