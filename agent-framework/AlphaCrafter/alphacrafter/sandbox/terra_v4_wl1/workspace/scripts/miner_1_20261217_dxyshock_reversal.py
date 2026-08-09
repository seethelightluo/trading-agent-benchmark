import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
ASOF=pd.Timestamp('2026-12-17')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root=Path('../persistent')
px={a:pd.read_csv(root/'stock_data'/f'{a}.csv',parse_dates=['date']).query('date <= @ASOF').set_index('date')['close'] for a in assets}
dxy=pd.read_csv(root/'index_data'/'DXY.csv',parse_dates=['date']).query('date <= @ASOF').set_index('date')['close']
close=pd.DataFrame(px).sort_index(); ret=close.pct_change(); dret=dxy.pct_change(5).reindex(close.index).ffill()
factor=-(ret.rolling(3).sum().shift(1)).mul(1+0.75*np.maximum(0,dret.shift(1)),axis=0)
fwd=ret.shift(-1); rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
ics=np.array([r[1] for r in rows]); print('dates',len(rows),'avg_n',np.mean([r[2] for r in rows]),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-17')]:
 q=[r[1] for r in rows if r[0]>=pd.Timestamp(lo) and r[0]<=pd.Timestamp(hi)]; print(lo,hi,len(q),np.mean(q) if q else np.nan)
print('coverage',factor.notna().mean().mean(),'turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',factor.index.min().date(),factor.index.max().date())
factor.reset_index().to_csv('scripts/miner_1_20261217_dxyshock_reversal_signal.csv',index=False)
