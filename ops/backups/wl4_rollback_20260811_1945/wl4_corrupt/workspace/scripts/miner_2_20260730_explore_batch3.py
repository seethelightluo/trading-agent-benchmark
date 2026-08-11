"""miner_2 batch-3 factor exploration v2 (optimized, vectorized autocorr).

Families:
  A. Relative strength vs cross-asset peers + trend acceleration / consistency
  B. Risk-adjusted return (Sharpe, Sortino), vectorized return autocorrelation
  C. Intraday vs overnight return split (uses open)
  D. Garman-Klass / Parkinson vol and vol ratios (uses high/low)
  E. Cross-asset conditional betas anchored on TRADABLE assets (SPX, XAU, BTC, US10Y)
  F. Volume-weighted momentum, OBV z-score

Validation window: 2020-01-01 .. 2026-07-15 (warm-up, no lookahead).
Admission gates (15-asset universe, horizon 10): |IC| >= 0.0070, |ICIR| >= 0.0840.
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


def load_asset(sym):
    df = get_stock_daily_data(symbol=sym, days=3000)
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
print(f"assets loaded: {sum(1 for v in frames.values() if v is not None)}/{len(WL)}", flush=True)

mac = {}
for m in ['VIX', 'DXY']:
    d = pd.read_csv(f"../persistent/index_data/{m}.csv")
    d['date'] = pd.to_datetime(d['date'])
    d = d[d['date'] <= END].set_index('date')
    d['close'] = pd.to_numeric(d['close'], errors='coerce')
    mac[m] = d['close'].dropna()

idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values() if f is not None])))
closes = pd.DataFrame({s: frames[s]['close'].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
opens = pd.DataFrame({s: frames[s]['open'].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
highs = pd.DataFrame({s: frames[s]['high'].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
lows = pd.DataFrame({s: frames[s]['low'].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
volume = pd.DataFrame({s: frames[s]['volume'].reindex(idx) for s in WL if frames[s] is not None}, index=idx)
rets = closes.pct_change()


def clean_panel(func):
    out = {}
    for a in WL:
        s = closes[a].dropna()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        try:
            out[a] = func(s).reindex(idx)
        except Exception:
            out[a] = pd.Series(np.nan, index=idx)
    return pd.DataFrame(out, index=idx)


def clean_rets(func):
    out = {}
    for a in WL:
        s = closes[a].dropna().pct_change()
        if len(s) < 30:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        try:
            out[a] = func(s).reindex(idx)
        except Exception:
            out[a] = pd.Series(np.nan, index=idx)
    return pd.DataFrame(out, index=idx)


def fwd_panel(h):
    out = {}
    for a in WL:
        s = closes[a].dropna()
        out[a] = (s.shift(-h) / s - 1.0).reindex(idx)
    return pd.DataFrame(out, index=idx)


FWD = {h: fwd_panel(h) for h in (1, 3, 5, 10, 20)}
C = {}

# ---------- A. relative strength & trend structure ----------
mom20 = clean_panel(lambda s: s / s.shift(20) - 1.0)
mom60 = clean_panel(lambda s: s / s.shift(60) - 1.0)
C['rel_mom_20d'] = mom20.sub(mom20.mean(axis=1), axis=0)
C['rel_mom_60d'] = mom60.sub(mom60.mean(axis=1), axis=0)
C['accel_60_20'] = clean_panel(lambda s: (s / s.shift(60) - 1.0) - (s / s.shift(20) - 1.0))
C['trend_consist_60'] = clean_rets(lambda r: (r > 0).rolling(60).mean() * 2.0 - 1.0)
C['updown_ratio_60'] = clean_rets(lambda r: (r.clip(lower=0).rolling(60).sum().abs())
                                  / (r.clip(upper=0).rolling(60).sum().abs() + 1e-9))
# ---------- B. risk-adjusted return & autocorr (vectorized) ----------
C['sharpe_60d'] = clean_rets(lambda r: r.rolling(60).mean() / r.rolling(60).std())
C['sortino_60d'] = clean_rets(lambda r: r.rolling(60).mean() / np.sqrt((r.clip(upper=0) ** 2).rolling(60).mean()))
C['autocorr_20d'] = clean_rets(lambda r: (r * r.shift(1)).rolling(20).sum()
                               / (r.rolling(20).std() * r.shift(1).rolling(20).std() + 1e-12))
C['autocorr_60d'] = clean_rets(lambda r: (r * r.shift(1)).rolling(60).sum()
                               / (r.rolling(60).std() * r.shift(1).rolling(60).std() + 1e-12))
# ---------- C. intraday / overnight ----------
C['intraday_eff_20'] = clean_panel(lambda s: ((s / opens[a].reindex(s.index) - 1.0).rolling(20).sum()
                                              / ((s / opens[a].reindex(s.index) - 1.0).abs().rolling(20).sum() + 1e-12)))
C['overnight_mom_20'] = clean_panel(lambda s: (opens[a].reindex(s.index) / s.shift(1) - 1.0).rolling(20).sum())
C['intraday_mom_20'] = clean_panel(lambda s: (s / opens[a].reindex(s.index) - 1.0).rolling(20).sum())
# ---------- D. Garman-Klass / Parkinson ----------
C['gk_vol_20'] = clean_panel(lambda s: np.sqrt((0.5 * np.log(highs[a].reindex(s.index) / lows[a].reindex(s.index)) ** 2
                                                - (2 * np.log(2) - 1) * np.log(opens[a].reindex(s.index) / s.shift(1)) ** 2)
                                               .rolling(20).mean().clip(lower=0)))
C['park_vol_20'] = clean_panel(lambda s: np.sqrt((np.log(highs[a].reindex(s.index) / lows[a].reindex(s.index)) ** 2)
                                                 .rolling(20).mean() / (4 * np.log(2))))
C['gk_cc_vol_ratio_20'] = C['gk_vol_20'] / (rets.rolling(20).std() + 1e-12)
C['down_up_vol_ratio_20'] = clean_rets(lambda r: r.clip(upper=0).rolling(20).std()
                                       / (r.clip(lower=0).rolling(20).std() + 1e-12))
# ---------- E. conditional betas ----------
def cond_beta_series(anchor_sym, sign, win=60, cond_mom=20):
    anc = closes[anchor_sym].dropna()
    anc_ret = anc.pct_change()
    anc_mom = anc_ret.rolling(cond_mom).mean() * cond_mom
    out = {}
    for a in WL:
        s = closes[a].dropna()
        d = pd.concat([s.pct_change().rename('a'), anc_ret.rename('c')], axis=1).dropna()
        if len(d) < 120:
            out[a] = pd.Series(np.nan, index=idx)
            continue
        beta = d['a'].rolling(win).cov(d['c']) / d['c'].rolling(win).var().replace(0, np.nan)
        out[a] = (sign * beta * anc_mom.reindex(beta.index)).reindex(idx)
    return pd.DataFrame(out, index=idx)


C['eq_beta_spx_60x20'] = cond_beta_series('SPX', 1.0)
C['safehav_beta_xau_60x20'] = cond_beta_series('XAU', 1.0)
C['crypto_beta_btc_60x20'] = cond_beta_series('BTC', 1.0)
C['rate_beta_us10y_60x20'] = cond_beta_series('US10Y', -1.0)
C['defensive_spx_60x20'] = cond_beta_series('SPX', -1.0)

# ---------- F. volume-weighted ----------
C['vw_mom_20d'] = clean_panel(lambda s: ((volume[a].reindex(s.index).fillna(0.0) * s.pct_change()).rolling(20).sum()
                                         / (volume[a].reindex(s.index).fillna(0.0).rolling(20).sum() + 1e-9)))
C['obv_z_60d'] = clean_panel(lambda s: ((np.sign(s.diff()) * volume[a].reindex(s.index).fillna(0.0)).cumsum()
                                        .pipe(lambda o: (o - o.rolling(60).mean()) / (o.rolling(60).std() + 1e-12))))


# ---------- library signals (provenance only) ----------
LIB = {}
LIB['mom_10d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(15) - 1.0)
LIB['mom_120d_skip5'] = clean_panel(lambda s: s.shift(5) / s.shift(125) - 1.0)
LIB['vol_of_vol20x60'] = clean_rets(lambda r: r.rolling(20).std().rolling(60).std())
vix_anc = mac['VIX']
vix_ret = vix_anc.pct_change()
vix_mom = vix_ret.rolling(20).mean() * 20.0
vix_lib = {}
for a in WL:
    s = closes[a].dropna()
    d = pd.concat([s.pct_change().rename('a'), vix_ret.reindex(s.index).dropna().rename('c')], axis=1).dropna()
    if len(d) < 120:
        vix_lib[a] = pd.Series(np.nan, index=idx)
        continue
    beta = d['a'].rolling(60).cov(d['c']) / d['c'].rolling(60).var().replace(0, np.nan)
    vix_lib[a] = (-1.0 * beta * vix_mom.reindex(beta.index)).reindex(idx)
LIB['vix_beta_cond_60x20'] = pd.DataFrame(vix_lib, index=idx)


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


results = []
for name, fp in C.items():
    ic = ic_series(fp)
    if len(ic) < 200:
        results.append(dict(name=name, n_ic=len(ic), note='insufficient IC dates'))
        print(f"{name:26s} NOTE insufficient", flush=True)
        continue
    icir = ic.mean() / ic.std(ddof=1)
    hit = float((np.sign(ic) == np.sign(ic.mean())).mean())
    valid = fp.notna()
    cov_ad = float(valid.sum().sum() / (len(fp) * len(fp.columns)))
    wd = fp.index.dayofweek < 5
    cov_d8 = float(((valid.sum(axis=1) >= MIN_ASSETS) & wd).mean())
    ranks = fp.rank(axis=1)
    to = float((ranks - ranks.shift(10)).abs().mean().mean())
    rho, rkey = max_lib_corr(fp)
    res = dict(name=name, ic=round(float(ic.mean()), 4), icir=round(float(icir), 4),
               hit=round(hit, 3), n_ic=len(ic), cov_ad=round(cov_ad, 3), cov_d8=round(cov_d8, 3),
               to=round(to, 3), rho=round(rho, 3), rkey=rkey)
    ok = abs(res['ic']) >= 0.007 and abs(res['icir']) >= 0.084
    print(f"{name:26s} IC={res['ic']:>8.4f} ICIR={res['icir']:>8.4f} hit={res['hit']:.3f} n={res['n_ic']:5d} "
          f"covAD={res['cov_ad']:.3f} covD8={res['cov_d8']:.3f} to={res['to']:.3f} "
          f"rho={res['rho']:.3f}({res['rkey']}) {'PASS' if ok else ''}", flush=True)
    results.append(res)

print("\n--- sorted by |ICIR| ---", flush=True)
for r in sorted(results, key=lambda x: -abs(x.get('icir', 0))):
    if 'note' in r:
        continue
    ok = abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084
    print(f"{'PASS' if ok else '    '} {r['name']:26s} IC={r['ic']:>8.4f} ICIR={r['icir']:>8.4f} "
          f"n={r['n_ic']:5d} covAD={r['cov_ad']:.3f} rho={r['rho']:.3f}({r['rkey']})", flush=True)

# ---------- decay for passers ----------
print("\n--- decay profile (passers) ---", flush=True)
for r in sorted(results, key=lambda x: -abs(x.get('icir', 0))):
    if 'note' in r:
        continue
    ok = abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084
    if ok:
        fp = C[r['name']]
        dec = {}
        for h in (1, 3, 5, 10, 20):
            s = ic_series(fp, h)
            if len(s) >= 100:
                dec[str(h)] = round(float(s.mean()), 4)
        print(f"{r['name']:26s} decay={dec}", flush=True)
