import pandas as pd, numpy as np, glob

NAMES = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
VIS = '2032-04-05'

def load(n):
    df = pd.read_csv(f'../persistent/stock_data/{n}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[df['date'] <= VIS].set_index('date')

closes = {n: load(n)['close'].astype(float) for n in NAMES}
vols   = {n: load(n)['volume'].astype(float) for n in NAMES}

grid = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
R = pd.DataFrame({n: closes[n].reindex(grid).pct_change() for n in NAMES})
print("grid dates:", len(grid), grid[0].date(), "->", grid[-1].date())

# ---- candidate factors ----
F = {}

# 1) Skewness 20d (skip 5): right-skewed assets may continue
for n in NAMES:
    s = closes[n]
    p = s.pct_change()
    F.setdefault('skew_20d_skip5',{})[n] = (p.shift(5).rolling(20).skew()).reindex(grid)

# 2) Drawdown recovery: current price vs rolling 60d max (how far below peak)
for n in NAMES:
    s = closes[n]
    F.setdefault('drawdown_60d',{})[n] = ((s / s.rolling(60).max()) - 1.0).reindex(grid)

# 3) Return asymmetry / semi-upside minus downside
for n in NAMES:
    p = closes[n].pct_change()
    up = p.where(p>0,0.0)
    dn = p.where(p<0,0.0)
    F.setdefault('updown_ratio_20',{})[n] = ((up.rolling(20).sum())/( -dn.rolling(20).sum()+1e-12)).reindex(grid)

# 4) 5d momentum (skip 1)
for n in NAMES:
    s = closes[n]
    F.setdefault('mom_5d_skip1',{})[n] = (s.shift(1)/s.shift(6)-1).reindex(grid)

# 5) 60d momentum (skip 20)
for n in NAMES:
    s = closes[n]
    F.setdefault('mom_60d_skip20',{})[n] = (s.shift(20)/s.shift(80)-1).reindex(grid)

# 6) Volatility trend: RV20 / RV60 (rising/falling vol regime)
for n in NAMES:
    p = closes[n].pct_change()
    F.setdefault('vol_ratio_20x60',{})[n] = (p.rolling(20).std()/p.rolling(60).std()).reindex(grid)

# 7) Cross momentum of crypto lead: BTC/ETH 5d lead vs rest
for n in NAMES:
    if n in ('BTC','ETH'):
        continue
    run = closes[n].pct_change().rolling(20).mean()
    F.setdefault('crypto_lead_20',{})[n] = run.reindex(grid)

ow = R.mean(axis=1, skipna=True).rolling(20).mean().reindex(grid)  # EW market trend
for n in NAMES:
    if n in ('BTC','ETH'):   # seed crypto lead from market to keep coverage
        F['crypto_lead_20'][n] = ow.reindex(grid)
    run = closes[n].pct_change().rolling(20).mean()
    F.setdefault('rel_vs_ew_20',{})[n] = (run - ow).reindex(grid)

# ---- rank IC at horizon 10, full window + recent 1y/2y ----
def rank_ic(fid, horizon=10, min_cov=8):
    FF = pd.DataFrame(F[fid])
    fwd = pd.DataFrame({n: closes[n].reindex(grid).shift(-horizon)/closes[n].reindex(grid)-1.0 for n in NAMES})
    ics = []
    dates = []
    for t in range(len(grid)-horizon):
        frow = FF.iloc[t]; rrow = fwd.iloc[t]
        m = frow.notna() & rrow.notna()
        if m.sum() < min_cov: continue
        ic = frow[m].rank().corr(rrow[m].rank())
        if pd.notna(ic): ics.append(ic); dates.append(grid[t])
    return np.array(ics), np.array(dates)

print(f"\n{'factor':<20}{'fullIC':>8}{'fullICIR':>9}{'1yIC':>7}{'1yICIR':>8}{'2yIC':>7}{'2yICIR':>8}{'n':>6}")
for fid in F:
    ics, dates = rank_ic(fid)
    if len(ics)==0:
        print(f"{fid:<20} no data"); continue
    d1 = np.array([d>=np.datetime64('2031-04-05') for d in dates])
    d2 = np.array([d>=np.datetime64('2030-04-05') for d in dates])
    s1, s2 = ics[d1], ics[d2]
    def ir(x): return x.mean()/x.std() if len(x)>1 and x.std()>0 else 0.0
    print(f"{fid:<20}{ics.mean():>8.4f}{ir(ics):>9.3f}"
          f"{s1.mean() if len(s1) else 0:>7.4f}{ir(s1) if len(s1) else 0:>8.3f}"
          f"{s2.mean() if len(s2) else 0:>7.4f}{ir(s2) if len(s2) else 0:>8.3f}{len(ics):>6}")