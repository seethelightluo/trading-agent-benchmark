import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=3000)
            if x is not None and len(x)>100:return x
        except Exception: pass
    return None
def main():
    ds={s:fetch(s) for s in U}; ds={s:x for s,x in ds.items() if x is not None}
    close=pd.DataFrame({s:x.set_index('date')['close'] for s,x in ds.items()}).sort_index(); r=close.pct_change()
    # Downside semideviation (zero for non-down days), with lagged ten-day reversal.
    down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=20).mean())*np.sqrt(252)
    fac=(-(close/close.shift(10)-1)/(down+0.005)).shift(1)
    print('symbols',len(ds),'dates',len(close),'cutoff',close.index.max().date())
    for h in [5,10,20]:
        fw=close.shift(-h)/close-1; rows=[]
        for d in fac.index:
            z=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
            if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
        a=pd.DataFrame(rows,columns=['date','ic','n']); a['date']=pd.to_datetime(a['date'])
        print('h',h,'dates',len(a),'avgN',round(a.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
        print('regimes',[(int(y),round(a[a.date.dt.year==y].ic.mean(),6),len(a[a.date.dt.year==y])) for y in sorted(a.date.dt.year.unique())[-5:]])
        if h==20:
            ranks=fac.rank(axis=1,pct=True); print('coverage',round(fac.notna().mean().mean(),6),'rank_turnover',round(ranks.diff().abs().mean(axis=1).dropna().mean(),6))
            fac.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20271021_downside_vol_reversal_signal.csv',index=False)
if __name__=='__main__': main()
