import pandas as pd, numpy as np

NAMES = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
VIS = '2030-10-07'

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

def load_idx(n):
    df = pd.read_csv(f'../persistent/index_data/{n}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df[df['date'] <= VIS].sort_values('date').set_index('date')['close'].astype(float)

jpy = load_idx('USDJPY'); jpy_r = jpy.pct_change(); jpy_20 = jpy/jpy.shift(20)-1
vix = load_idx('VIX'); vix_r = vix.pct_change(); vix_20 = vix/vix.shift(20)-1
dxy = load_idx('DXY')
eur = load_idx('EURUSD')

F = {}
for n in NAMES:
    s = closes[n]
    pct = s.pct_change()
    # 1) USDJPY beta conditional on USDJPY 20d move
    own = s.index
    jp_o = jpy_r.reindex(own).ffill(); jp20_o = jpy_20.reindex(own).ffill()
    b = pct.rolling(60).cov(jp_o)/jp_o.rolling(60).var()
    F.setdefault('jpy_beta_cond_60x20',{})[n] = (-b*jp20_o).reindex(grid)
    # 2) volatility reversal: recent vol (5d) relative to long vol (60d), negative
    rv5 = pct.rolling(5).std(); rv60 = pct.rolling(60).std()
    F.setdefault('vol_reversal_5x60',{})[n] = (-(rv5/rv60)).reindex(grid)
    # 3) drawdown depth (distance from 60d high) - mean reversion long
    F.setdefault('dd_60d',{})[n] = (s/s.rolling(60).max()-1.0).reindex(grid)
    # 4) risk-adjusted momentum (sharpe-like): demeaned return / vol over 40d
    mom40 = s/s.shift(40)-1.0
    F.setdefault('risk_adj_mom_40',{})[n] = (mom40/rv60.abs()).reindex(grid)
    # 5) VIX regime cross: 20d return when VIX falling (risk-on proxy)
    vix20_o = vix_20.reindex(own).ffill()
    F.setdefault('vix_risk_on_20x60',{})[n] = (-vix20_o*pct.rolling(60).sum()).reindex(grid)
    # 6) 5d reversal (short-term reversal after skip 1)
    F.setdefault('rev_5d_skip1',{})[n] = (-(s.shift(1)/s.shift(6)-1.0)).reindex(grid)
    # 7) upper shadow / range position factor
    hi = s.rolling(20).max(); lo = s.rolling(20).min()
    F.setdefault('range_pos_20',{})[n] = ((s-lo)/(hi-lo)).reindex(grid)

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

print(f"current date {VIS}, grid n={len(grid)}")
print(f"{'factor':<26}{'full_ic':>8}{'full_icir':>9}{'last60_ic':>11}{'last120_ic':>11}{'n':>6}")
for fid in sorted(F.keys()):
    ic = rank_ic(fid)
    if len(ic)==0:
        print(f"{fid:<26} no data"); continue
    icir = ic.mean()/ic.std() if ic.std()>0 else 0
    print(f"{fid:<26}{ic.mean():>8.4f}{icir:>9.4f}{ic[-60:].mean():>11.4f}{ic[-120:].mean():>11.4f}{len(ic):>6}")

print("\nAdmission gates: |IC|>=0.0070 and |ICIR|>=0.0840")
