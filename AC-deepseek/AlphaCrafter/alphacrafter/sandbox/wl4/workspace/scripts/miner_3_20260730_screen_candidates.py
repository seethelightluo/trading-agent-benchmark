"""miner_3 candidate factor screening on the 15-asset cross-asset universe.
Validation window: 2020-01-01 .. 2026-07-15 (research warm-up, consistent with library).
Metrics: daily cross-sectional Spearman rank IC vs fwd 10d return, ICIR, hit ratio,
coverage, turnover (mean abs rank change over 10d), max abs library correlation.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

WL = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
      'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
END = pd.Timestamp('2026-07-15')
HORIZON = 10
MIN_ASSETS = 8

# ---------------- data ----------------
def load_asset(sym):
    df = get_stock_daily_data(symbol=sym, days=4000)
    if df is None:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['ret'] = df['close'].pct_change()
    return df

def load_macro(name):
    p = f"../persistent/index_data/{name}.csv"
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['ret'] = df['close'].pct_change()
    return df[['close', 'ret']]

frames = {s: load_asset(s) for s in WL}
closes = pd.DataFrame({s: f['close'] for s, f in frames.items()}).dropna(how='all')
rets = closes.pct_change()
vols = {s: f['volume'] for s, f in frames.items()}
volume = pd.DataFrame(vols).reindex(closes.index)
macro = {m: load_macro(m) for m in ['VIX', 'DXY', 'USDJPY', 'EURUSD', 'USDCNY']}
vix_ret = macro['VIX']['ret'].reindex(closes.index)
vix = macro['VIX']['close'].reindex(closes.index)
dxy = macro['DXY']['close'].reindex(closes.index)
usd = macro['USDJPY']['close'].reindex(closes.index)

print(f"universe: {len(closes.columns)} assets, dates {closes.index.min().date()}..{closes.index.max().date()}, n={len(closes)}")

# forward return (t -> t+HORIZON)
fwd = closes.shift(-HORIZON) / closes - 1.0

# ---------------- library factor panels ----------------
def build_library():
    lib = {}
    lib['mom_10d_skip5'] = closes.shift(5) / closes.shift(15) - 1.0
    lib['mom_120d_skip5'] = closes.shift(5) / closes.shift(125) - 1.0
    lib['vol_of_vol20x60'] = rets.rolling(20).std().rolling(60).std()
    # vix beta conditional: -beta(asset, vix, 60) * (vix/vix.shift(20)-1)
    b = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
    for s in closes.columns:
        d = pd.concat([rets[s].rename('a'), vix_ret.rename('v')], axis=1).dropna()
        cov = d['a'].rolling(60).cov(d['v'])
        var = d['v'].rolling(60).var()
        beta = (cov / var.replace(0, np.nan)).reindex(closes.index)
        b[s] = -beta * (vix / vix.shift(20) - 1.0)
    lib['vix_beta_cond_60x20'] = b
    return lib

LIB = build_library()

# ---------------- analysis ----------------
def ic_series(factor_panel):
    out = {}
    idx = factor_panel.index
    for t in idx:
        fv = factor_panel.loc[t]
        fr = fwd.loc[t].reindex(fv.index)
        m = fv.notna() & fr.notna()
        if m.sum() < MIN_ASSETS:
            continue
        ic, _ = spearmanr(fv[m], fr[m])
        if np.isfinite(ic):
            out[t] = ic
    return pd.Series(out)

def analyze(name, factor_panel):
    if isinstance(factor_panel, pd.Series):
        factor_panel = factor_panel.to_frame(name='f')
    ic = ic_series(factor_panel)
    if len(ic) < 120:
        return dict(name=name, n_ic=len(ic), note='insufficient dates')
    icir = ic.mean() / ic.std()
    hit = float((np.sign(ic) == np.sign(ic.mean())).mean())
    # coverage
    valid = factor_panel.notna()
    cov_asset_days = float(valid.sum().sum() / (len(factor_panel) * len(factor_panel.columns)))
    cov_dates_ge8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    # turnover: mean abs rank change over 10 trading days
    ranks = factor_panel.rank(axis=1)
    tr = (ranks - ranks.shift(10)).abs().mean().mean()
    # library correlation
    rho = []
    for lid, lpanel in LIB.items():
        rr = []
        for t in ic.index:
            fv = factor_panel.loc[t]; lv = lpanel.loc[t].reindex(fv.index)
            m = fv.notna() & lv.notna()
            if m.sum() < MIN_ASSETS:
                continue
            r, _ = spearmanr(fv[m], lv[m])
            if np.isfinite(r):
                rr.append(r)
        rho.append(np.mean(rr) if rr else np.nan)
    max_rho = max((abs(r) for r in rho if np.isfinite(r)), default=0.0)
    return dict(name=name, ic=round(float(ic.mean()), 4), icir=round(float(icir), 4),
                hit=round(hit, 3), n_ic=len(ic), cov_asset_days=round(cov_asset_days, 3),
                cov_dates_ge8=round(cov_dates_ge8, 3), turnover=round(float(tr), 3),
                max_rho=round(float(max_rho), 3))

# ---------------- candidate factors ----------------
C = {}
C['eff_ratio_20d'] = (closes - closes.shift(20)).abs() / rets.abs().rolling(20).sum()
C['eff_ratio_60d'] = (closes - closes.shift(60)).abs() / rets.abs().rolling(60).sum()
C['range_pos_10d'] = (closes - closes.rolling(10).min()) / (closes.rolling(10).max() - closes.rolling(10).min())
C['range_pos_20d'] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())
C['w52high_prox'] = closes / closes.rolling(252).max() - 1.0
C['drawdown_prox_60d'] = closes / closes.rolling(60).max() - 1.0
C['vol_adj_mom_60d'] = (closes / closes.shift(60) - 1.0) / rets.rolling(20).std()
C['vol_adj_mom_20d'] = (closes / closes.shift(20) - 1.0) / rets.rolling(20).std()
C['amihud_20d'] = (rets.abs() / (volume + 1.0)).rolling(20).mean()
C['volume_trend_20_60'] = volume.rolling(20).mean() / volume.rolling(60).mean()
C['downside_semidev_60d'] = rets.clip(upper=0).rolling(60).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
C['skew_20d'] = rets.rolling(20).skew()
C['hl_range_20d'] = (closes.rolling(20).max() - closes.rolling(20).min()) / closes
C['overnight_gap_20d'] = (pd.concat([f['open'] / f['close'].shift(1) - 1 for f in frames.values()], axis=1).reindex(closes.index)).rolling(20).mean()
C['yld_spread_mom10d'] = None  # computed below
# macro-conditioned candidates (use observation-only signals)
C['dxy_beta_cond_60x20'] = None
C['usdjpy_beta_cond_60x20'] = None

# yield spread momentum: US10Y - CN10Y (levels are yields)
us10 = closes['US10Y']; cn10 = closes['CN10Y']
spread = us10 - cn10
C['yld_spread_mom10d'] = (spread - spread.shift(10)).reindex(closes.index).abs()  # abs change magnitude
C['yld_spread_level'] = spread.reindex(closes.index)

def cond_beta(cond_ret, name):
    b = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=float)
    for s in closes.columns:
        d = pd.concat([rets[s].rename('a'), cond_ret.rename('c')], axis=1).dropna()
        cov = d['a'].rolling(60).cov(d['c'])
        var = d['c'].rolling(60).var()
        beta = (cov / var.replace(0, np.nan)).reindex(closes.index)
        b[s] = -beta * (cond_ret / cond_ret.shift(20) - 1.0) if name == 'dxy' else beta * (cond_ret / cond_ret.shift(20) - 1.0)
    return b

C['dxy_beta_cond_60x20'] = cond_beta(macro['DXY']['ret'].reindex(closes.index), 'dxy')
C['usdjpy_beta_cond_60x20'] = cond_beta(macro['USDJPY']['ret'].reindex(closes.index), 'jpy')

results = []
for name, panel in C.items():
    if panel is None:
        continue
    res = analyze(name, panel)
    results.append(res)
    if 'note' in res:
        print(f"{name:24s} NOTE={res['note']}")
        continue
    print(f"{name:24s} IC={res.get('ic'):>8} ICIR={res.get('icir'):>8} hit={res.get('hit')} n={res.get('n_ic')} "
          f"covAD={res.get('cov_asset_days')} covD8={res.get('cov_dates_ge8')} to={res.get('turnover')} rho={res.get('max_rho')}")

print("\n--- summary sorted by |ICIR| ---")
for r in sorted(results, key=lambda x: -abs(x.get('icir', 0))):
    print(r)
