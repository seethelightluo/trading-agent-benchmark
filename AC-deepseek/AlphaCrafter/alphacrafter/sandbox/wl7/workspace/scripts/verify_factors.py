import pandas as pd, numpy as np

NAMES = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
VIS = '2026-07-29'

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

F = {}
for n in NAMES:
    s = closes[n]
    F.setdefault('mom_10d_skip5',{})[n] = (s.shift(5)/s.shift(15)-1).reindex(grid)
    F.setdefault('mom_120d_skip5',{})[n] = (s.shift(5)/s.shift(125)-1).reindex(grid)
    F.setdefault('rel_mom_raw',{})[n] = (s.shift(5)/s.shift(25)-1).reindex(grid)
    pct = s.pct_change()
    rv20 = pct.rolling(20).std()
    F.setdefault('vol_of_vol20x60',{})[n] = rv20.rolling(60).std().reindex(grid)
    neg = pct.where(pct<0,0.0)
    F.setdefault('downside_vol_ratio_20',{})[n] = (-(neg.rolling(20).std()/rv20)).reindex(grid)
    F.setdefault('max_ret_20d',{})[n] = pct.rolling(20).max().reindex(grid)
    av = (pct.abs()/vols[n].replace(0,np.nan)).rolling(20,min_periods=10).mean()
    F.setdefault('amihud_20',{})[n] = av.reindex(grid)

for n in NAMES:
    own = closes[n].index
    mkt = R.loc[own].mean(axis=1, skipna=True)
    r = closes[n].pct_change()
    cov = r.rolling(60).cov(mkt); var = mkt.rolling(60).var()
    F.setdefault('beta_ew_60d',{})[n] = (cov/var).reindex(grid)

vdf = pd.read_csv('../persistent/index_data/VIX.csv')
vdf.columns = [c.strip().lower() for c in vdf.columns]
vdf['date'] = pd.to_datetime(vdf['date'])
vix = vdf[vdf['date'] <= VIS].sort_values('date').set_index('date')['close'].astype(float)
vixr = vix.pct_change()
vix_up = (vix/vix.shift(20)-1.0)
for n in NAMES:
    own = closes[n].index
    v_own = vixr.reindex(own).ffill()
    v_up_own = vix_up.reindex(own).ffill()
    r = closes[n].pct_change()
    b = r.rolling(60).cov(v_own)/v_own.rolling(60).var()
    F.setdefault('vix_beta_cond_60x20',{})[n] = (-b*v_up_own).reindex(grid)

def rank_ic(fid, horizon=10, min_cov=8):
    FF = pd.DataFrame(F[fid])
    fwd = pd.DataFrame({n: closes[n].reindex(grid).shift(-horizon)/closes[n].reindex(grid)-1.0 for n in NAMES})
    ics=[]
    for t in range(len(grid)-horizon):
        frow=FF.iloc[t]; rrow=fwd.iloc[t]
        m=frow.notna() & rrow.notna()
        if m.sum()<min_cov: continue
        ic=frow[m].rank().corr(rrow[m].rank())
        if pd.notna(ic): ics.append(ic)
    return np.array(ics)

print(f"{'factor':<22}{'last30':>8}{'last60':>8}{'last120':>9}{'full':>8}{'n':>6}")
for fid in ['rel_mom_raw','mom_120d_skip5','beta_ew_60d','vol_of_vol20x60','vix_beta_cond_60x20','amihud_20','mom_10d_skip5','downside_vol_ratio_20','max_ret_20d']:
    ic = rank_ic(fid)
    if len(ic)==0:
        print(f"{fid:<22} no data"); continue
    print(f"{fid:<22}{ic[-30:].mean():>8.4f}{ic[-60:].mean():>8.4f}{ic[-120:].mean():>9.4f}{ic.mean():>8.4f}{len(ic):>6}")

print("\nLatest ranks (2026-07-29), selected factors:")
sel = ['rel_mom_raw','mom_120d_skip5','beta_ew_60d','vol_of_vol20x60','vix_beta_cond_60x20','amihud_20','mom_10d_skip5']
for fid in sel:
    FF = pd.DataFrame(F[fid])
    row = FF.iloc[-1]
    print(f"{fid:<22}", row.dropna().rank(pct=True).round(2).sort_values(ascending=False).to_dict())
