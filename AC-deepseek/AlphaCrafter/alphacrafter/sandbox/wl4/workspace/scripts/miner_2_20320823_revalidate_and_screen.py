"""MINER_2 2032-08-23: re-validate 3 existing effective factors + screen new
candidate factors on the 15-asset cross-asset universe. Data only through
visible_through = 2032-08-20 (from ../persistent/date.json). No live-account
interaction. Admission gates: |IC|>=0.007, |ICIR|>=0.084 (daily rank IC, h=10).
"""
import json, os, math
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
    """rolling beta of y on x over window, min_obs valid pairs"""
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

def factor_skew(win=60):
    return ret.rolling(win).skew()

def factor_beta_macro(macro_col, win=60, min_obs=40):
    x = mac[macro_col].pct_change()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

def factor_range_pos(win=20):
    rng = (hi - lo).replace(0, np.nan)
    pos = (px - lo) / rng
    return pos.rolling(win).mean()

def factor_amihud(win=20):
    illiq = (ret.abs() / vol.replace(0, np.nan))
    return illiq.rolling(win).mean()

def factor_max_ret(win=20):
    return ret.rolling(win).max()

def factor_min_ret(win=20):
    return ret.rolling(win).min()

def factor_rate_spread_beta(win=60, min_obs=40):
    spread = px["US10Y"] - px["CN10Y"]
    x = spread.diff()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

def factor_vix_beta(win=60, min_obs=40):
    x = mac["VIX"].pct_change()
    return pd.DataFrame({a: rolling_beta(ret[a], x, win, min_obs) for a in ASSETS})

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
    ic = rank_ic_series(sig)
    ic = ic.dropna()
    if len(ic) < 20:
        print(f"{name:26s} INSUFFICIENT IC dates: {len(ic)}")
        return None
    m = ic.mean(); sd = ic.std(ddof=1)
    icir = m / sd if sd > 0 else np.nan
    hit = (ic > 0).mean()
    cov_ad = sig.notna().sum().sum() / (len(sig) * len(ASSETS))
    cov_dates_ge8 = (sig.notna().sum(axis=1) >= MIN_VALID).mean()
    # turnover: mean abs rank change over 10d
    rk = sig.rank(axis=1)
    turn = rk.diff(10).abs().mean().mean()
    # decay
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
    # sub-periods
    sub = {}
    if len(ic) >= 60:
        for label, n in [("r250", 250), ("r500", 500), ("r750", 750)]:
            s2 = ic.iloc[-min(n, len(ic)):]
            m2 = s2.mean(); sd2 = s2.std(ddof=1) if len(s2) > 2 else np.nan
            sub[label] = {"ic": float(m2), "icir": float(m2 / sd2) if sd2 and sd2 > 0 else np.nan,
                          "n": len(s2)}
    # yearly IC
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
    "skew_60d": factor_skew(60),
    "dxy_beta_60d": factor_beta_macro("DXY"),
    "usdjpy_beta_60d": factor_beta_macro("USDJPY"),
    "vix_beta_60d": factor_vix_beta(),
    "rate_spread_beta_60d": factor_rate_spread_beta(),
    "range_pos_20d": factor_range_pos(20),
    "amihud_illiq_20d": factor_amihud(20),
    "max_ret_20d": factor_max_ret(20),
    "min_ret_20d": factor_min_ret(20),
}
cand_res = {}
for name, sig in candidates.items():
    cand_res[name] = validate(name, sig)

json.dump({"existing": existing_res, "candidates": cand_res, "visible": VISIBLE},
          open("logs/miner2_20320823_screen.json", "w"), indent=1, default=str)
print("\nsaved logs/miner2_20320823_screen.json")
