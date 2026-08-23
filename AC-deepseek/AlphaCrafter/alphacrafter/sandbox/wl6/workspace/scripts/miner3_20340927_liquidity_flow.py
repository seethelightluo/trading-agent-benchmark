import pandas as pd, numpy as np, os, base64, zlib, hashlib, json

CUR = pd.Timestamp('2034-09-20')
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
DATA = '../persistent/stock_data'

def load(a):
    df = pd.read_csv(os.path.join(DATA, a + '.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUR].sort_values('date').reset_index(drop=True)
    return df

panels = {a: load(a) for a in ASSETS}

# Build date union (index), long-only forward returns held in dict
# Use full cross-asset alignment via outer union on dates. Core asset 4050-4156 dates, BTC/ETH go further back.
all_dates = sorted(set.union(*[set(panels[a]['date']) for a in ASSETS]))
all_dates = [pd.Timestamp(d) for d in all_dates]
date_idx = {d: i for i, d in enumerate(all_dates)}
N = len(all_dates)
print('total union dates', N, 'range', all_dates[0].date(), '..', all_dates[-1].date())

close = pd.DataFrame(index=all_dates, columns=ASSETS, dtype=float)
for a in ASSETS:
    df = panels[a]
    close.loc[df['date'].values, a] = df['close'].values

# start analysis at 2020-06-01 to allow warmup
start = pd.Timestamp('2020-06-01')
sid = date_idx[start]
close = close.iloc[sid:]  # slice rows
all_dates = close.index.tolist()
date_idx = {d: i for i, d in enumerate(all_dates)}

ret = close.pct_change()

# ---------- factor: liquidity-adjusted momentum proxy using volume participation ----------
# vol_ma20 = rolling mean volume
# vol_pct = volume / rolling_ma(volume, 60)
# liquidity factor: 1 - |1 - vol_pct|  -> high when volume near trend (stable liquidity), low when volume spike/dearth
# combined with momentum proxy: mom_10 = close/close.shift(10)-1
# factor = mom10 * liq_stability  (concentrated tidal momentum: trend amplified by stable liquidity)

C = close
V = pd.DataFrame(index=all_dates, columns=ASSETS, dtype=float)
for a in ASSETS:
    df = panels[a]
    s = df.set_index('date')['volume']
    s = s.reindex(all_dates).astype(float)
    V[a] = s.values

vol_ma60 = V.rolling(60, min_periods=30).mean()
vol_pct = V / vol_ma60
liq_stab = 1.0 - (vol_pct - 1.0).abs()   # high when volume near its recent trend
mom10 = C / C.shift(10) - 1.0

# abs-return weighted to catch realized-return participation vs volume spikes
absret = ret.abs()
# volume-return consistency: if big volume + small move => churn/trap; if big volume + big move => real trend
vol_ret_ratio = V / (absret.replace(0, np.nan).rolling(20, min_periods=10).mean() * C)

F = mom10 * liq_stab
F2 = mom10 * np.sign(liq_stab - liq_stab.rolling(120, min_periods=60).mean())   # mom where liq stability is above its own history
F3 = -vol_pct.shift(1).rank(axis=1)  # low volume-participation = high liquidity efficiency (crowded->mean reversion) direction

HORIZON = 10

def rank_ic_series(Fact, hor=HORIZON):
    # Fact rows indexed by date
    f = Fact.shift(1)          # use factor known at t to predict ret from t..t+hor
    fwd = close.shift(-hor) / close - 1.0
    ics, dates = [], []
    for i in range(len(close)):
        dt = all_dates[i]
        frow = f.iloc[i]
        rrow = fwd.iloc[i]
        m = frow.notna() & rrow.notna()
        if m.sum() >= 8:
            ic = np.corrcoef(frow[m], rrow[m])[0, 1]
            if np.isfinite(ic):
                ics.append(ic); dates.append(dt)
    return np.array(ics), dates

def report(name, Fact):
    ics, dates = rank_ic_series(Fact)
    ic = ics.mean() if len(ics) else np.nan
    std = ics.std(ddof=1) if len(ics) > 1 else np.nan
    icir = ic / std if std and not np.isnan(std) else np.nan
    hit = (np.sign(ics) == np.sign(ic)).mean() if len(ics) and not np.isnan(ic) else np.nan
    cov_asset_days = float(Fact.notna().mean().mean())
    cov_dates_ge8 = np.mean([ (Fact.iloc[i].notna().sum() >= 8) for i in range(len(Fact)) ])
    Fr = Fact.iloc[enumerate_dates(dates)].rank(axis=1)
    to = float(Fr.diff().abs().mean().mean()) if len(dates) > 1 else np.nan
    print('%-28s ic=%.4f icir=%.4f hit=%.3f n=%d cov_ad=%.3f cov_d8=%.3f to=%.4f | ic>=.007:%s icir>=.084:%s' % (
        name, ic, icir, hit, len(ics), cov_asset_days, cov_dates_ge8, to,
        (abs(ic) >= 0.0070), (abs(icir) >= 0.0840)))
    return dict(ic=ic, icir=icir, hit=hit, n=len(ics), cov_ad=cov_asset_days, cov_d8=cov_dates_ge8, to=to)

def enumerate_dates(dates):
    locs = [date_idx[d] for d in dates]
    idx = pd.RangeIndex(len(close))
    return idx.isin(pd.Index(locs))

print('\n--- Factor validation (horizon=%d) ---' % HORIZON)
r1 = report('liq_mom_tidal(mom10*liqstab)', F)
r2 = report('liq_mom_relstability(mom10*sign)', F2)
r3 = report('liq_efficiency(-vol_pct_rank)', F3)