"""MINER_2 2032-09-20: re-validate 3 existing effective factors + screen new
candidate factors on the 15-asset cross-asset universe. Data only through
visible_through (from ../persistent/date.json). No live-account interaction.
Admission gates: |IC|>=0.007, |ICIR|>=0.084 (daily rank IC, h=10).
"""
import json, os
import pandas as pd
import numpy as np

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"
VISIBLE = json.load(open("../persistent/date.json"))["visible_through"]
HORIZON = 10
MIN_VALID = 8

# ---------------- data ----------------
px_all = {}
for a in ASSETS:
    df = pd.read_csv(os.path.join(DATA_DIR, a + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    px_all[a] = df.set_index("date")["close"].astype(float)
px = pd.DataFrame(px_all).sort_index()

hi = lo = op = vol = {}
for a in ASSETS:
    df = pd.read_csv(os.path.join(DATA_DIR, a + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].set_index("date")
    hi[a] = df["high"]; lo[a] = df["low"]; op[a] = df["open"]; vol[a] = df["volume"]
hi = pd.DataFrame(hi).sort_index(); lo = pd.DataFrame(lo).sort_index()
op = pd.DataFrame(op).sort_index(); vol = pd.DataFrame(vol).sort_index()

mac = {}
for f in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    df = pd.read_csv(os.path.join(IDX_DIR, f + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    mac[f] = df.set_index("date")["close"].astype(float)
mac = pd.DataFrame(mac).sort_index()

ret = px.pct_change()
mkt = ret.mean(axis=1)

print(f"panel dates: {px.index.min().date()} -> {px.index.max().date()} n={len(px)}")
print(f"macro dates: {mac.index.min().date()} -> {mac.index.max().date()} n={len(mac)}")

# ---------------- factor builders ----------------
def rolling_beta(y, x, win=60, min_obs=40):
    out = np.full(len(px), np.nan)
    xv, yv = x.values, y.values
    for i in range(win, len(px)):
        sx, sy = xv[i-win:i], yv[i-win:i]
        m = ~(np.isnan(sx) | np.isnan(sy))
        if m.sum() >= min_obs:
            sx, sy = sx[m], sy[m]
            vx = np.var(sx)
            if vx > 1e-12:
                out[i] = np.cov(sx, sy)[0, 1] / vx
    return pd.Series(out, index=px.index)

def factor_dn_mkt_beta(win=60, min_obs=40):
    down = mkt.where(mkt < 0)
    return pd.DataFrame({a: rolling_beta(ret[a], down, win, min_obs) for a in ASSETS})

def factor_rate_beta_cn10y(win=60, min_obs=40):
    x = px["CN10Y"].pct_change()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

def factor_vol_adj_mom_accel(fast=20, slow=60, vwin=20):
    mom_f = px / px.shift(fast) - 1.0
    mom_s = px / px.shift(slow) - 1.0
    v = ret.rolling(vwin).std()
    return (mom_f - mom_s) / v

def factor_beta_macro(macro_col, win=60, min_obs=40):
    x = mac[macro_col].pct_change()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

def factor_beta_asset(asset, win=60, min_obs=40):
    x = px[asset].pct_change()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

def factor_overnight_ratio(win=20):
    # cumulative overnight (gap) return / cumulative total return over win
    gap = op / px.shift(1) - 1.0
    tot = px / px.shift(win) - 1.0
    gap_cum = (1 + gap).rolling(win).apply(np.prod, raw=True) - 1.0
    return gap_cum / tot.replace(0, np.nan)

def factor_kaufman_eff(win=20):
    path = ret.abs().rolling(win).sum()
    net = (px / px.shift(win) - 1.0).abs()
    return net / path.replace(0, np.nan)

def factor_adx(win=14):
    prev_close = px.shift()
    tr = pd.concat([(hi - lo), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1)
    tr = tr.groupby(level=0).max()
    up = hi.diff()
    dn = -lo.diff()
    plus_dm = pd.DataFrame(np.where((up > dn) & (up > 0), up, 0.0), index=px.index, columns=ASSETS)
    minus_dm = pd.DataFrame(np.where((dn > up) & (dn > 0), dn, 0.0), index=px.index, columns=ASSETS)
    tr_s = tr.rolling(win).mean().replace(0, np.nan)
    pdi = 100 * plus_dm.rolling(win).mean() / tr_s
    mdi = 100 * minus_dm.rolling(win).mean() / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(win).mean()

def factor_upper_shadow(win=20):
    body_hi = pd.concat([op, px], axis=1).groupby(level=0).max()
    rng = (hi - lo).replace(0, np.nan)
    sh = (hi - body_hi) / rng
    return sh.rolling(win).mean()

def factor_lower_shadow(win=20):
    body_lo = pd.concat([op, px], axis=1).groupby(level=0).min()
    rng = (hi - lo).replace(0, np.nan)
    sh = (body_lo - lo) / rng
    return sh.rolling(win).mean()

def factor_tail_ratio(win=60):
    up = ret.rolling(win).quantile(0.95)
    dn = ret.rolling(win).quantile(0.05).abs()
    return up / dn.replace(0, np.nan)

def factor_bull_bear_vol(win=60):
    upm = ret.where(ret > 0).rolling(win).mean()
    dnm = ret.where(ret < 0).rolling(win).mean().abs()
    return upm / dnm.replace(0, np.nan)

def factor_skew(win=20):
    return ret.rolling(win).skew()

def factor_max_gain_loss(win=20):
    mx = ret.rolling(win).max()
    mn = ret.rolling(win).min().abs()
    return mx / mn.replace(0, np.nan)

def factor_consistency(win=60):
    # risk-adjusted mean return (m/s) times excess up-day fraction (2*frac_up-1)
    m = ret.rolling(win).mean()
    s = ret.rolling(win).std().replace(0, np.nan)
    frac_up = (ret > 0).rolling(win).mean()
    return (m / s) * (2 * frac_up - 1)

# ---------------- validation ----------------
fwd = px.shift(-HORIZON) / px - 1.0

def rank_ic_series(sig):
    ic = []
    for dt in sig.index:
        s, f = sig.loc[dt], fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= MIN_VALID:
            ic.append((dt, np.corrcoef(s[m].rank(), f[m].rank())[0, 1]))
    return pd.Series(dict(ic)).sort_index()

def validate(name, sig, full=True):
    ic = rank_ic_series(sig).dropna()
    if len(ic) < 20:
        print(f"{name:26s} INSUFFICIENT IC dates: {len(ic)}")
        return None
    m = ic.mean(); sd = ic.std(ddof=1)
    icir = m / sd if sd > 0 else np.nan
    hit = (ic > 0).mean()
    cov_ad = sig.notna().sum().sum() / (len(sig) * len(ASSETS))
    cov_dates_ge8 = (sig.notna().sum(axis=1) >= MIN_VALID).mean()
    rk = sig.rank(axis=1)
    turn = rk.diff(10).abs().mean().mean()
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fh = px.shift(-h) / px - 1.0
        ics = []
        for dt in sig.index:
            s, f = sig.loc[dt], fh.loc[dt]
            mm = s.notna() & f.notna()
            if mm.sum() >= MIN_VALID:
                ics.append(np.corrcoef(s[mm].rank(), f[mm].rank())[0, 1])
        decay[h] = float(np.nanmean(ics)) if ics else np.nan
    sub = {}
    if len(ic) >= 60:
        for label, n in [("r250", 250), ("r500", 500), ("r750", 750)]:
            s2 = ic.iloc[-min(n, len(ic)):]
            m2 = s2.mean(); sd2 = s2.std(ddof=1) if len(s2) > 2 else np.nan
            sub[label] = {"ic": float(m2), "icir": float(m2 / sd2) if sd2 and sd2 > 0 else np.nan,
                          "n": len(s2)}
    yearly = ic.groupby(ic.index.year).mean().to_dict()
    res = {"name": name, "ic": float(m), "icir": float(icir), "hit": float(hit),
           "n_ic_dates": len(ic), "ic_std": float(sd), "cov_asset_days": float(cov_ad),
           "cov_dates_ge8": float(cov_dates_ge8), "turnover_10d_rank": float(turn),
           "decay": {str(k): round(v, 4) for k, v in decay.items()},
           "sub": sub, "yearly_ic": {str(k): round(v, 4) for k, v in yearly.items()}}
    if full:
        print(f"\n=== {name} ===")
        print(f"  IC={m:+.4f} ICIR={icir:+.3f} hit={hit:.2f} n={len(ic)} std={sd:.3f}")
        print(f"  coverage: asset_days={cov_ad:.3f} dates_ge8={cov_dates_ge8:.3f} turnover10={turn:.2f}")
        print(f"  decay: {res['decay']}")
        print(f"  sub-periods: { {k: (round(v['ic'],4), round(v['icir'],3)) for k,v in sub.items()} }")
        print(f"  yearly IC: {res['yearly_ic']}")
    return res

def lib_corr(cand_sig, name, lib):
    """max abs pairwise signal correlation vs library factors (rank-level, aligned dates)."""
    best = None
    for lid, lsig in lib.items():
        df = pd.concat([cand_sig.rank(axis=1).stack(), lsig.rank(axis=1).stack()], axis=1, keys=["c", "l"])
        df = df.dropna()
        if len(df) < 100:
            continue
        rho = np.corrcoef(df["c"], df["l"])[0, 1]
        if best is None or abs(rho) > abs(best[1]):
            best = (lid, rho)
    return best

print("\n" + "="*100)
print("PART A: RE-VALIDATION OF EXISTING EFFECTIVE FACTORS (through", VISIBLE, ")")
print("="*100)
existing = {
    "vol_adj_mom_accel_20x60": factor_vol_adj_mom_accel(),
    "dn_mkt_beta_60d": factor_dn_mkt_beta(),
    "rate_beta_cn10y_60d": factor_rate_beta_cn10y(),
}
existing_res = {}
for name, sig in existing.items():
    existing_res[name] = validate(name, sig)

print("\n" + "="*100)
print("PART B: NEW CANDIDATE SCREEN (through", VISIBLE, ")")
print("="*100)
candidates = {
    "overnight_ratio_20d": factor_overnight_ratio(20),
    "kaufman_eff_20d": factor_kaufman_eff(20),
    "adx_14d": factor_adx(14),
    "upper_shadow_20d": factor_upper_shadow(20),
    "lower_shadow_20d": factor_lower_shadow(20),
    "tail_ratio_60d": factor_tail_ratio(60),
    "bull_bear_vol_60d": factor_bull_bear_vol(60),
    "skew_20d": factor_skew(20),
    "max_gain_loss_20d": factor_max_gain_loss(20),
    "consistency_60d": factor_consistency(60),
    "wti_beta_60d": factor_beta_asset("WTI"),
    "xau_beta_60d": factor_beta_asset("XAU"),
    "btc_beta_60d": factor_beta_asset("BTC"),
    "dxy_beta_60d": factor_beta_macro("DXY"),
    "vix_beta_60d": factor_beta_macro("VIX"),
}
cand_res = {}
for name, sig in candidates.items():
    r = validate(name, sig)
    if r is not None:
        best = lib_corr(sig, name, existing)
        r["max_abs_library_correlation"] = round(abs(best[1]), 4) if best else None
        r["max_corr_factor"] = best[0] if best else None
        if best:
            print(f"  [lib-corr] max_abs_library_correlation={abs(best[1]):.3f} vs {best[0]}")
    cand_res[name] = r

os.makedirs("logs", exist_ok=True)
json.dump({"existing": existing_res, "candidates": cand_res, "visible": VISIBLE},
          open("logs/miner2_20320920_screen.json", "w"), indent=1, default=str)
print("\nsaved logs/miner2_20320920_screen.json")
