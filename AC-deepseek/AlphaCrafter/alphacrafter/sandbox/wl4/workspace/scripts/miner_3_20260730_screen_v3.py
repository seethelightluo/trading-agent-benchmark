"""miner_3 candidate factor screening v3 - fully per-asset clean pipeline.

Root-cause fix: BTC/ETH trade 7d/week while equity/commodity indices trade 5d/week.
Any rolling/shift window applied on a union calendar spans weekend NaN rows and
returns NaN (min_periods=window). v3 computes EVERY factor and forward return on
each asset's OWN calendar (dropna'd), then aligns only for the cross-sectional IC.

Universe: 15 tradable cross-asset instruments. Validation window 2020-01-01..2026-07-15.
Admission gates (15-asset universe): |IC| >= 0.0070, |ICIR| >= 0.0840 at horizon 10.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WL = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
END = pd.Timestamp('2026-07-15')
MIN_ASSETS = 8
HORIZON = 10

# ---------------- data loading: per-asset own calendar ----------------
def load_asset(sym):
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[df['close'].notna()]

frames = {s: load_asset(s) for s in WL}
print(f"assets loaded: {sum(1 for v in frames.values() if v is not None)}/{len(WL)}")
for s in WL:
    f = frames[s]
    if f is not None:
        print(f"  {s:10s} rows={len(f):5d} {f.index.min().date()}..{f.index.max().date()} "
              f"weekend={int((f.index.dayofweek >= 5).sum())} vol_nonnull={int(f['volume'].notna().sum())}")

# macro signals (5d/week own calendars)
mac = {}
for m in ['VIX', 'DXY', 'USDJPY', 'EURUSD', 'USDCNY']:
    p = f"../persistent/index_data/{m}.csv"
    d = pd.read_csv(p)
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['date'] <= END].set_index('date')
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    mac[m] = d['close'].dropna()
    mac[m + '_ret'] = mac[m].pct_change()

# ---------------- per-asset factor & forward return ----------------
def own(frames_a, cols=None):
    """per-asset DataFrame aligned on union index (NaN where asset has no data)."""
    idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames_a.values() if f is not None])))
    return pd.DataFrame({s: (frames_a[s][cols] if cols else frames_a[s]).reindex(idx)
                         for s in frames_a if frames_a[s] is not None})

closes = own(frames, 'close')
rets = closes.pct_change()
opens = own(frames, 'open')
volume = own(frames, 'volume')

def clean_panel(func):
    """Apply func to each asset's own dropna'd series, return union-indexed panel."""
    idx = closes.index
    out = {}
    for a in WL:
        s = closes[a].dropna()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        r = func(s)
        out[a] = r.reindex(idx)
    return pd.DataFrame(out, index=idx)

def clean_rets(func):
    idx = closes.index
    out = {}
    for a in WL:
        s = closes[a].dropna().pct_change()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        r = func(s)
        out[a] = r.reindex(idx)
    return pd.DataFrame(out, index=idx)

def fwd_panel(h):
    """Per-asset forward h-day return on own calendar."""
    idx = closes.index
    out = {}
    for a in WL:
        s = closes[a].dropna()
        r = s.shift(-h) / s - 1.0
        out[a] = r.reindex(idx)
    return pd.DataFrame(out, index=idx)

FWD = {h: fwd_panel(h) for h in (1, 2, 3, 5, 10, 20)}

# ---------------- candidates (all per-asset clean) ----------------
C = {}
# momentum / trend
C['mom_20d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(25) - 1.0)
C['mom_60d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(65) - 1.0)
C['eff_ratio_20d'] = clean_panel(lambda s: (s - s.shift(20)).abs() / s.pct_change().abs().rolling(20).sum())
C['eff_ratio_60d'] = clean_panel(lambda s: (s - s.shift(60)).abs() / s.pct_change().abs().rolling(60).sum())
C['sma_slope_60_20'] = clean_panel(lambda s: s.rolling(60).mean() / s.rolling(60).mean().shift(20) - 1.0)
C['rsi_14'] = clean_panel(lambda s: 100 - 100 / (1 + s.pct_change().clip(lower=0).rolling(14).mean()
                                                 / (s.pct_change().clip(upper=0).abs().rolling(14).mean() + 1e-9)))
# mean reversion / range
C['range_pos_10d'] = clean_panel(lambda s: (s - s.rolling(10).min()) / (s.rolling(10).max() - s.rolling(10).min()))
C['range_pos_20d'] = clean_panel(lambda s: (s - s.rolling(20).min()) / (s.rolling(20).max() - s.rolling(20).min()))
C['dd_prox_60d'] = clean_panel(lambda s: s / s.rolling(60).max() - 1.0)
C['w52high_prox'] = clean_panel(lambda s: s / s.rolling(252).max() - 1.0)
C['bollinger_z_20'] = clean_panel(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
# volatility / risk
C['vol_adj_mom_20d'] = clean_panel(lambda s: (s / s.shift(20) - 1.0) / s.pct_change().rolling(20).std())
C['vol_adj_mom_60d'] = clean_panel(lambda s: (s / s.shift(60) - 1.0) / s.pct_change().rolling(60).std())
C['vol_ratio_10_60'] = clean_rets(lambda r: r.rolling(10).std() / r.rolling(60).std())
C['vol_zscore_1y'] = clean_rets(lambda r: (r.rolling(20).std() - r.rolling(20).std().rolling(252).mean())
                                / r.rolling(20).std().rolling(252).std())
C['downside_sd_60d'] = clean_rets(lambda r: np.sqrt((r.clip(upper=0) ** 2).rolling(60).mean()))
C['skew_20d'] = clean_rets(lambda r: r.rolling(20).skew())
C['updown_ratio_20d'] = clean_rets(lambda r: r.clip(lower=0).rolling(20).sum().abs()
                                   / (r.clip(upper=0).rolling(20).sum().abs() + 1e-9))
# liquidity / volume
def amihud_panel(win=20):
    idx = closes.index
    out = {}
    for a in WL:
        s = closes[a].dropna()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        v = volume[a].reindex(s.index).fillna(0.0)
        am = (s.pct_change().abs() / (v + 1.0)).rolling(win).mean()
        out[a] = am.reindex(idx)
    return pd.DataFrame(out, index=idx)

C['amihud_20d'] = amihud_panel(20)
C['vol_trend_20_60'] = clean_panel(lambda s: s.rolling(20).mean() / s.rolling(60).mean())

# macro-beta conditional factors (own-calendar beta then align)
def cond_beta(cond, cond_ret, sign, win=60):
    idx = closes.index
    out = {}
    for a in WL:
        s = closes[a].dropna()
        if len(s) < 120:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        cr = cond_ret.reindex(s.index).dropna()
        d = pd.concat([s.pct_change().rename('a'), cr.rename('c')], axis=1).dropna()
        if len(d) < 120:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        beta = (d['a'].rolling(win).cov(d['c']) / d['c'].rolling(win).var().replace(0, np.nan))
        condm = (cond / cond.shift(20) - 1.0).reindex(beta.index)
        out[a] = (sign * beta * condm).reindex(idx)
    return pd.DataFrame(out, index=idx)

C['vix_hedge_cond_60x20'] = cond_beta(mac['VIX'], mac['VIX_ret'], 1.0)
C['dxy_beta_cond_60x20'] = cond_beta(mac['DXY'], mac['DXY_ret'], -1.0)
C['usdjpy_beta_cond_60x20'] = cond_beta(mac['USDJPY'], mac['USDJPY_ret'], 1.0)
C['eurusd_beta_cond_60x20'] = cond_beta(mac['EURUSD'], mac['EURUSD_ret'], -1.0)

# systemic-risk: rolling avg pairwise correlation (own-calendar, monthly recompute)
def cross_corr(win=60, step=5):
    idx = closes.index
    out = pd.DataFrame(index=idx, columns=WL, dtype=float)
    r = rets[WL]
    last = None
    for i, t in enumerate(idx):
        if i % step != 0 and last is not None:
            out.loc[t] = last
            continue
        w = r.loc[:t].tail(win)
        w = w.dropna(how='all')
        if len(w) < 30:
            continue
        c = w.corr()
        row = {}
        for a in WL:
            if a in c.index:
                others = c[a].drop(index=a, errors='ignore')
                row[a] = others.mean()
        last = pd.Series(row)
        out.loc[t] = last
    return out

C['cross_corr_60d'] = cross_corr(60)

# ---------------- library signals (per-asset clean) ----------------
LIB = {}
LIB['mom_10d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(15) - 1.0)
LIB['mom_120d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(125) - 1.0)
LIB['vol_of_vol20x60'] = clean_rets(lambda r: r.rolling(20).std().rolling(60).std())
LIB['vix_beta_cond_60x20'] = cond_beta(mac['VIX'], mac['VIX_ret'], -1.0)  # matches library def

# ---------------- IC machinery ----------------
def ic_series(fp, h=HORIZON):
    fwd = FWD[h]
    out = {}
    for t in fp.index:
        if t.dayofweek >= 5:
            continue
        fv = fp.loc[t]
        fr = fwd.loc[t].reindex(fv.index)
        m = fv.notna() & fr.notna()
        if int(m.sum()) < MIN_ASSETS:
            continue
        ic, _ = spearmanr(fv[m], fr[m])
        if np.isfinite(ic):
            out[t] = ic
    return pd.Series(out)

def max_lib_corr(fp):
    best, best_key = 0.0, None
    for lid, lp in LIB.items():
        both = pd.concat([fp.stack().rename('c'), lp.stack().rename('l')], axis=1).dropna()
        if len(both) < 300:
            continue
        r = float(both['c'].corr(both['l']))
        if abs(r) > best:
            best, best_key = abs(r), lid
    return best, best_key

def analyze(name, fp):
    ic = ic_series(fp)
    if len(ic) < 200:
        return dict(name=name, n_ic=len(ic), note='insufficient IC dates')
    icir = ic.mean() / ic.std(ddof=1)
    hit = float((np.sign(ic) == np.sign(ic.mean())).mean())
    valid = fp.notna()
    cov_ad = float(valid.sum().sum() / (len(fp) * len(fp.columns)))
    wd = fp.index.dayofweek < 5
    cov_d8 = float(((valid.sum(axis=1) >= MIN_ASSETS) & wd).mean())
    ranks = fp.rank(axis=1)
    to = float((ranks - ranks.shift(10)).abs().mean().mean())
    rho, rkey = max_lib_corr(fp)
    decay = {str(h): round(float(ic_series(fp, h).mean()), 4) for h in (1, 2, 3, 5, 10, 20)
             if len(ic_series(fp, h)) >= 100}
    return dict(name=name, ic=round(float(ic.mean()), 4), icir=round(float(icir), 4),
                hit=round(hit, 3), n_ic=len(ic), cov_ad=round(cov_ad, 3), cov_d8=round(cov_d8, 3),
                to=round(to, 3), rho=round(rho, 3), rkey=rkey, decay=decay)

results = []
for name, fp in C.items():
    res = analyze(name, fp)
    results.append(res)
    if 'note' in res:
        print(f"{name:26s} NOTE={res['note']}")
        continue
    ok = abs(res['ic']) >= 0.007 and abs(res['icir']) >= 0.084
    print(f"{name:26s} IC={res['ic']:>8.4f} ICIR={res['icir']:>8.4f} hit={res['hit']:.3f} n={res['n_ic']:5d} "
          f"covAD={res['cov_ad']:.3f} covD8={res['cov_d8']:.3f} to={res['to']:.3f} "
          f"rho={res['rho']:.3f}({res['rkey']}) {'PASS' if ok else ''}")

print("\n--- candidates sorted by |ICIR| (PASS gate |IC|>=0.007 & |ICIR|>=0.084) ---")
for r in sorted(results, key=lambda x: -abs(x.get('icir', 0))):
    if 'note' in r:
        continue
    ok = abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084
    print(f"{'PASS' if ok else '    '} {r['name']:26s} IC={r['ic']:>8.4f} ICIR={r['icir']:>8.4f} "
          f"n={r['n_ic']:5d} covAD={r['cov_ad']:.3f} rho={r['rho']:.3f}({r['rkey']}) decay={r['decay']}")
