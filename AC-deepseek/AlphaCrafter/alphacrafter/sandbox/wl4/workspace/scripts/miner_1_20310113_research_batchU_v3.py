"""MINER_1 2031-01-13: cross-asset factor exploration + re-validation (batch U, v3).

Visible data through 2031-01-10 (previous completed trading day before current
date 2031-01-13). No live-account interaction. Screens candidate factors with
rank IC vs forward-10d returns across the 15-instrument cross-asset universe
(min_valid=8). Admission gates: |IC|>=0.0070, |ICIR|>=0.0840 (same-horizon h=10).
Fixes: NaN-aware beta via explicit pairwise masking (loop over windows).
"""
import json, os
import numpy as np
import pandas as pd

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"
VISIBLE = "2031-01-10"

# ---------- load panels ----------
def load_close(symbol, ddir):
    df = pd.read_csv(os.path.join(ddir, symbol + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    return df.set_index("date")["close"].astype(float)

px = pd.DataFrame({a: load_close(a, DATA_DIR) for a in ASSETS}).sort_index()
print("panel:", px.index.min().date(), "->", px.index.max().date(), "n=", len(px))
ret = px.pct_change()
mkt = ret.mean(axis=1)

def load_ohlc(symbol, ddir):
    df = pd.read_csv(os.path.join(ddir, symbol + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    return df.set_index("date")[["open","high","low","close"]].astype(float)

ohlc = {a: load_ohlc(a, DATA_DIR) for a in ASSETS}

macro = {}
for m in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    s = load_close(m, IDX_DIR)
    macro[m] = s.reindex(px.index)
macro_df = pd.DataFrame(macro)

# ---------- helpers ----------
def rank_ic_series(factor_panel, fwd, min_valid=8):
    ic = {}
    for dt in factor_panel.index:
        s = factor_panel.loc[dt]
        f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= min_valid:
            ic[dt] = np.corrcoef(s[m].rank(), f[m].rank())[0, 1]
    return pd.Series(ic)

def summarize(ic):
    ic = ic.dropna()
    m = float(ic.mean())
    sd = float(ic.std(ddof=1)) if len(ic) > 2 else float("nan")
    icir = m / sd if sd and sd > 0 else float("nan")
    return {"ic": m, "icir": icir, "ic_hit": float((ic > 0).mean()), "n": len(ic),
            "ic_std": sd, "last_ic": float(ic.iloc[-1]) if len(ic) else float("nan")}

def recent_ic(ic, label, windows=(63, 126, 252, 504)):
    out = {}
    for w in windows:
        sub = ic.iloc[-w:]
        if len(sub) == 0:
            out[label + f"_{w}"] = (float("nan"), float("nan"))
            continue
        mm = float(sub.mean())
        ss = float(sub.std(ddof=1)) if len(sub) > 2 else float("nan")
        out[label + f"_{w}"] = (mm, mm / ss if ss and ss > 0 else float("nan"))
    return out

def coverage_assets(factor_panel):
    valid = factor_panel.notna().sum(axis=1)
    return float((valid >= 8).mean())

def turnover_rank(factor_panel, step=10):
    r = factor_panel.rank(axis=1)
    chg = (r - r.shift(step)).abs().mean(axis=1).dropna()
    return float(chg.mean())

# NaN-aware rolling beta (loop over windows, explicit pairwise masking)
def beta_to(x, win=60, min_obs=40):
    out = {}
    for a in ASSETS:
        y = ret[a]
        betas = np.full(len(px), np.nan)
        xv, yv = x.values, y.values
        for i in range(win, len(px)):
            sx, sy = xv[i-win:i], yv[i-win:i]
            m = ~(np.isnan(sx) | np.isnan(sy))
            if m.sum() >= min_obs:
                sxm, sym = sx[m], sy[m]
                vx = np.var(sxm)
                if vx > 1e-12:
                    betas[i] = np.cov(sxm, sym)[0, 1] / vx
        out[a] = pd.Series(betas, index=px.index)
    return pd.DataFrame(out)

# ---------- forward returns (h=10) ----------
fwd = px.shift(-10) / px - 1.0

# ---------- 1) re-validate current effective factors ----------
def dn_mkt_beta(win=60, min_obs=40):
    down = mkt.where(mkt < 0)
    return beta_to(down, win, min_obs)

def rate_beta(win=60, min_obs=40):
    cn10y = px["CN10Y"].pct_change()
    return beta_to(cn10y, win, min_obs)

def vol_adj_mom_accel(fast=20, slow=60, vwin=20):
    mom_f = px / px.shift(fast) - 1.0
    mom_s = px / px.shift(slow) - 1.0
    vol = ret.rolling(vwin).std()
    return (mom_f - mom_s) / vol

existing = {
    "vol_adj_mom_accel_20x60": vol_adj_mom_accel(),
    "dn_mkt_beta_60d": dn_mkt_beta(),
    "rate_beta_cn10y_60d": rate_beta(),
}
print("\n=== RE-VALIDATION of current effective factors (full + recent) ===")
for name, sig in existing.items():
    ics = rank_ic_series(sig, fwd)
    s = summarize(ics)
    r = recent_ic(ics, "r")
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit']:.2f} n={s['n']} "
          f"| r63=({r['r_63'][0]:+.3f},{r['r_63'][1]:+.2f}) r126=({r['r_126'][0]:+.3f},{r['r_126'][1]:+.2f}) "
          f"r252=({r['r_252'][0]:+.3f},{r['r_252'][1]:+.2f}) r504=({r['r_504'][0]:+.3f},{r['r_504'][1]:+.2f}) "
          f"cov={coverage_assets(sig):.2f} to={turnover_rank(sig):.2f}")

# ---------- 2) candidate factor screen ----------
print("\n=== CANDIDATE SCREEN (full history + recent) ===")
cands = {}

vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()

def parkinson_vol(win=20):
    out = {}
    for a in ASSETS:
        h = ohlc[a]["high"]; l = ohlc[a]["low"]
        hl = (np.log(h) - np.log(l)) ** 2
        out[a] = np.sqrt(hl.rolling(win).mean() / (4.0 * np.log(2.0)))
    return pd.DataFrame(out)

def garman_klass_vol(win=20):
    out = {}
    for a in ASSETS:
        o = ohlc[a]["open"]; h = ohlc[a]["high"]; l = ohlc[a]["low"]; c = ohlc[a]["close"]
        v = 0.5 * (np.log(h) - np.log(l)) ** 2 - (2.0 * np.log(2.0) - 1.0) * (np.log(c) - np.log(o)) ** 2
        out[a] = np.sqrt(v.rolling(win).mean())
    return pd.DataFrame(out)

pv20 = parkinson_vol(20)
cands["inv_parkinson_vol_20d"] = -pv20
cands["parkinson_ratio_20_60"] = parkinson_vol(20) / parkinson_vol(60)
cands["inv_gk_vol_20d"] = -garman_klass_vol(20)

def eff_ratio(win=20):
    num = (px - px.shift(win)).abs()
    den = ret.abs().rolling(win).sum()
    return num / den

cands["eff_ratio_20d"] = eff_ratio(20)
cands["eff_ratio_60d"] = eff_ratio(60)

ma20 = px.rolling(20).mean()
ma60 = px.rolling(60).mean()
cands["close_vs_ma20"] = px / ma20 - 1.0
cands["close_vs_ma60"] = px / ma60 - 1.0
cands["bollinger_pos_20"] = (px - ma20) / (2.0 * vol20)
cands["ma20_ma60_cross"] = (ma20 - ma60) / ma60

sharpe60 = ret.rolling(60).mean() / ret.rolling(60).std()
cands["sharpe_60d"] = sharpe60

up = ret.where(ret > 0, 0.0).rolling(60).std()
dn = ret.where(ret < 0, 0.0).rolling(60).std()
cands["semi_vol_ratio_60d"] = up / dn

corr_mkt = ret.rolling(60).corr(mkt)
cands["corr_mkt_60d"] = corr_mkt

comm_ret = ret[["XAU","COPPER","WTI"]].mean(axis=1)
cands["commodity_beta_60d"] = beta_to(comm_ret)
cands["btc_beta_60d"] = beta_to(ret["BTC"])
cands["us10y_beta_60d"] = beta_to(px["US10Y"].pct_change())

spread = px["US10Y"] - px["CN10Y"]
cands["spread_beta_60d"] = beta_to(spread.pct_change())

m20 = px / px.shift(20) - 1.0
lowvol = (vol20 < vol20.rolling(252).median()).astype(float)
cands["mom20_lowvol_cond"] = m20 * lowvol

vix_level = macro_df["VIX"]
calm = (vix_level < vix_level.rolling(252).median()).astype(float)
cands["inv_vol_calm_cond"] = -vol20 * calm

cands["dd_depth_60d"] = px / px.rolling(60).max() - 1.0
cands["dd_depth_120d"] = px / px.rolling(120).max() - 1.0

cands["winrate_60d"] = (ret > 0).rolling(60).mean()

def vol_trend(fast=5, slow=60):
    out = {}
    for a in ASSETS:
        df = pd.read_csv(os.path.join(DATA_DIR, a + ".csv"), parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE)]
        v = df.set_index("date")["volume"].astype(float).reindex(px.index)
        out[a] = v.rolling(fast).mean() / v.rolling(slow).mean()
    return pd.DataFrame(out)

cands["volume_trend_5_60"] = vol_trend(5, 60)

cands["close_vs_20d_high"] = px / px.rolling(20).max() - 1.0

def roll_skew(s, win=60):
    return s.rolling(win).apply(lambda w: pd.Series(w).skew() if len(w) >= 30 else np.nan, raw=False)

skew60 = ret.apply(lambda c: roll_skew(c, 60))
cands["ret_skew_60d"] = skew60

results = {}
for name, sig in cands.items():
    ics = rank_ic_series(sig, fwd)
    s = summarize(ics)
    r = recent_ic(ics, "r")
    results[name] = (s, r, sig)
    flag = ""
    if abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840:
        flag = "  <== FULL-PASS"
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit']:.2f} n={s['n']} "
          f"| r63=({r['r_63'][0]:+.3f},{r['r_63'][1]:+.2f}) r126=({r['r_126'][0]:+.3f},{r['r_126'][1]:+.2f}) "
          f"r252=({r['r_252'][0]:+.3f},{r['r_252'][1]:+.2f}) r504=({r['r_504'][0]:+.3f},{r['r_504'][1]:+.2f}) "
          f"cov={coverage_assets(sig):.2f} to={turnover_rank(sig):.2f}{flag}")

# ---------- 3) decay + library correlation for full-pass candidates ----------
print("\n=== DECAY (h=1,3,5,10,20) + LIBRARY CORRELATION for full-pass candidates ===")
lib = dict(existing)
lib["mom_10d_skip5"] = px.shift(5) / px.shift(15) - 1.0
lib["mom_120d_skip5"] = px.shift(5) / px.shift(125) - 1.0
lib["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()
vix_ret = macro_df["VIX"].pct_change()
vix_beta = beta_to(vix_ret)
lib["vix_beta_cond_60x20"] = -vix_beta * (macro_df["VIX"] / macro_df["VIX"].shift(20) - 1.0)

passing = {k: v for k, v in results.items()
           if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
for name, (s, r, sig) in passing.items():
    dec = {}
    for h in (1, 3, 5, 10, 20):
        fh = px.shift(-h) / px - 1.0
        ih = rank_ic_series(sig, fh)
        dec[str(h)] = round(float(ih.mean()), 4) if len(ih) else float("nan")
    best, key = 0.0, None
    for lname, lsig in lib.items():
        if lsig is None:
            continue
        both = pd.concat([sig.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"]))
        if abs(rr) > best:
            best, key = abs(rr), lname
    print(f"{name:26s} decay={dec} max_abs_lib_corr={best:.4f} (vs {key})")

print("\ndone")
