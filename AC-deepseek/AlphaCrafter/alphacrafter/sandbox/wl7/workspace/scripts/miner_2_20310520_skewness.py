import pandas as pd, numpy as np

NAMES = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
VIS = pd.Timestamp('2031-05-19')

def load(n):
    df = pd.read_csv(f'../persistent/stock_data/{n}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[df['date'] <= VIS].set_index('date')

closes = {n: load(n)['close'].astype(float) for n in NAMES}
grid = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
grid = grid[grid <= VIS]
print('grid dates <= vis:', len(grid))

def skew_series(s, window=20, min_periods=12):
    r = s.pct_change()
    return r.rolling(window, min_periods=min_periods).skew().reindex(grid)

F = {n: skew_series(closes[n]) for n in NAMES}
FF = pd.DataFrame(F)

def rank_ic(horizon=10, min_cov=8):
    fwd = pd.DataFrame({n: closes[n].reindex(grid).shift(-horizon)/closes[n].reindex(grid)-1.0 for n in NAMES})
    ics=[]
    for t in range(len(grid)-horizon):
        frow=FF.iloc[t]; rrow=fwd.iloc[t]
        m=frow.notna() & rrow.notna()
        if m.sum()<min_cov: continue
        ic=frow[m].rank().corr(rrow[m].rank())
        if pd.notna(ic): ics.append(ic)
    return np.array(ics)

for horizon in [5,10,20]:
    ic = rank_ic(horizon=horizon)
    if len(ic)==0:
        print(f'h={horizon:<3d} no data'); continue
    icir = ic.mean()/ic.std() if ic.std()>0 else np.nan
    print(f'skewness_20d_skip5 h={horizon:<3d} full_ic={ic.mean():.4f} icir={icir:.4f} hit={np.mean(ic>0):.3f} n={len(ic)} last30={ic[-30:].mean():.4f} last60={ic[-60:].mean():.4f} last120={ic[-120:].mean():.4f}')

ic10 = rank_ic(horizon=10)
icir10 = ic10.mean()/ic10.std()
print('\nGATE h=10: abs_ic=%.4f (need >=0.0070), abs_icir=%.4f (need >=0.0840)'%(abs(ic10.mean()), abs(icir10)))
print('direction sign:', 1 if ic10.mean()>0 else -1)
print('coverage_dates_ge8: %.3f'% np.mean([ FF.iloc[t].notna().sum()>=8 for t in range(len(grid)) ]))

idx = grid[:len(ic10)]
dfy = pd.DataFrame({'ic':ic10}, index=idx)
print('\nPer-year h10 IC:')
for yr, grp in dfy.groupby(dfy.index.year):
    print(f'  {yr}: ic={grp.ic.mean():.4f} icir={grp.ic.mean()/grp.ic.std():.4f} n={len(grp)}')