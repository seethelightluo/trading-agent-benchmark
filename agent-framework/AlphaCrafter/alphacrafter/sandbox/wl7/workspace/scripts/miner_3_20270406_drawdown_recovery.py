import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

# Drawdown-recovery momentum: lagged 30d return, penalized by depth of trailing 60d drawdown.
acct=get_account_dict(); syms=acct.get('watch_list',[])
if not syms: syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
    d=get_stock_daily_data(s, days=3000)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(frames).sort_index(); rets=prices.pct_change()
# factor known at t-1, aligned against return t (API history cutoff is naturally current)
rollmax=prices.rolling(60,min_periods=40).max()
dd=(prices/rollmax-1).rolling(60,min_periods=40).min().abs()
f=(prices.pct_change(30)/ (0.01+dd)).shift(1)
f=f.replace([np.inf,-np.inf],np.nan)
forward=prices.pct_change().shift(-1)
ics=[]; rows=[]
for dt in f.index:
    x=f.loc[dt]; y=forward.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
        if pd.notna(ic): ics.append(ic); rows.append((dt,ic,len(z)))
a=np.array(ics); print('symbols',len(frames),'dates',len(a),'avg_n',np.mean([r[2] for r in rows]))
print('daily_ic %.8f icir %.8f hit %.4f coverage %.4f' % (a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(1),np.mean(a>0), f.notna().sum().sum()/(len(f)*len(frames))))
# ICIR convention daily mean/std
for h in [5,10,20]:
    yy=prices.pct_change(h).shift(-h); aa=[]
    for dt in f.index:
      z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(z)>=8:
       q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
       if pd.notna(q): aa.append(q)
    print('decay',h,'%.8f %.8f'%(np.mean(aa),np.mean(aa)/(np.std(aa,ddof=1)+1e-12)))
for name,mask in [('2020-22',(pd.to_datetime([r[0] for r in rows]).year<=2022)),('2023-24',pd.to_datetime([r[0] for r in rows]).year.isin([2023,2024])),('2025-27',(pd.to_datetime([r[0] for r in rows]).year>=2025))]:
 q=a[mask]; print(name,len(q),'%.8f %.8f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)))
# artifact
out=pd.DataFrame(rows,columns=['date','signal_ic','n']); out.to_csv('scripts/miner_3_20270406_drawdown_recovery_signal.csv',index=False)
print('artifact scripts/miner_3_20270406_drawdown_recovery_signal.csv')
