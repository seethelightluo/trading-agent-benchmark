"""MINER_1 2030-12-02: cross-asset factor exploration + re-validation.

Visible data through 2030-11-29. No live-account interaction.
Screens candidate factors with rank IC vs forward-10d returns across the
15-instrument cross-asset universe. Admission gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json, os, math
import numpy as np
import pandas as pd

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"
VISIBLE = "2030-11-29"

# ---------- load panels ----------
def load_close(symbol, ddir):
    df = pd.read_csv(os.path.join(ddir, symbol + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    return df.set_index("date")["close"].astype(float)

px = pd.DataFrame({a: load_close(a, DATA_DIR) for a in ASSETS}).sort_index()
print("panel:", px.index.min().date(), "->", px.index.max().date(), "n=", len(px))
ret = px.pct_change()
mkt = ret.mean(axis=1)

macro = {}
for m in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
    s = load_close(m, IDX_DIR)
    macro[m] = s.reindex(px.index)
macro_df = pd.DataFrame(macro)
print("macro coverage last date:", macro_df.notna().iloc[-1].sum(), "/ 5")

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

def recent_ic(ic, label, windows=(63, 126, 252)):
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

# ---------- forward returns ----------
fwd = px.shift(-10) / px - 1.0  # forward 10d return, aligned so factor@t -> ret(t..t+10)

# ---------- 1) re-validate current effective factors ----------
def dn_mkt_beta(win=60, min_obs=40):
    down = mkt.where(mkt < 0)
    out = {}
    for a in ASSETS:
        x, y = down.values, ret[a].values
        betas = np.full(len(px), np.nan)
        for i in range(win, len(px)):
            m = ~(np.isnan(x[i-win:i]) | np.isnan(y[i-win:i]))
            if m.sum() >= min_obs:
                sx, sy = x[i-win:i][m], y[i-win:i][m]
                vx = np.var(sx)
                if vx > 1e-12:
                    betas[i] = np.cov(sx, sy)[0, 1] / vx
        out[a] = pd.Series(betas, index=px.index)
    return pd.DataFrame(out)

def rate_beta(win=60, min_obs=40):
    cn10y = px["CN10Y"].pct_change()
    out = {}
    for a in ASSETS:
        x, y = cn10y.values, ret[a].values
        betas = np.full(len(px), np.nan)
        for i in range(win, len(px)):
            m = ~(np.isnan(x[i-win:i]) | np.isnan(y[i-win:i]))
            if m.sum() >= min_obs:
                sx, sy = x[i-win:i][m], y[i-win:i][m]
                vx = np.var(sx)
                if vx > 1e-12:
                    betas[i] = np.cov(sx, sy)[0, 1] / vx
        out[a] = pd.Series(betas, index=px.index)
    return pd.DataFrame(out)

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
print("\n=== RE-VALIDATION of current effective factors (full history) ===")
recent_all = {}
for name, sig in existing.items():
    ics = rank_ic_series(sig, fwd)
    s = summarize(ics)
    r = recent_ic(ics, "r")
    recent_all[name] = (s, r)
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit']:.2f} n={s['n']} "
          f"| r63=({r['r_63'][0]:+.3f},{r['r_63'][1]:+.2f}) r126=({r['r_126'][0]:+.3f},{r['r_126'][1]:+.2f}) "
          f"r252=({r['r_252'][0]:+.3f},{r['r_252'][1]:+.2f}) cov={coverage_assets(sig):.2f} to={turnover_rank(sig):.2f}")

# ---------- 2) candidate factor screen ----------
print("\n=== CANDIDATE SCREEN (full history + recent) ===")
cands = {}

# USD beta 60d
def beta_to(x, win=60):
    out = {}
    for a in ASSETS:
        z = pd.concat([ret[a].rename("a"), x.rename("x")], axis=1)
        b = z["a"].rolling(win).cov(z["x"]) / z["x"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out)

cands["usd_beta_60d"] = beta_to(macro_df["DXY"].pct_change())
cands["usdjpy_beta_60d"] = beta_to(macro_df["USDJPY"].pct_change())
cands["vix_beta_plain_60d"] = beta_to(macro_df["VIX"].pct_change())
cands["eurusd_beta_60d"] = beta_to(macro_df["EURUSD"].pct_change())

# equity basket beta (equal-weight of the 8 equity names)
eq_names = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX"]
eq_ret = ret[eq_names].mean(axis=1)
cands["equity_beta_60d"] = beta_to(eq_ret)

# trend / drawdown
cands["dd_from_high_60d"] = px / px.rolling(60).max() - 1.0
cands["dd_from_high_120d"] = px / px.rolling(120).max() - 1.0
cands["range_pos_20d"] = (px - px.rolling(20).min()) / (px.rolling(20).max() - px.rolling(20).min())

# volatility / low-vol
vol20 = ret.rolling(20).std()
cands["inv_vol_20d"] = -vol20
cands["range_20d_norm"] = (px.rolling(20).max() - px.rolling(20).min()) / px

# return shape
def roll_skew(s, win=20):
    return s.rolling(win).apply(lambda w: pd.Series(w).skew() if len(w) >= 12 else np.nan, raw=False)

skew20 = ret.apply(lambda c: roll_skew(c, 20))
cands["ret_skew_20d"] = skew20

def roll_kurt(s, win=20):
    return s.rolling(win).apply(lambda w: pd.Series(w).kurt() if len(w) >= 12 else np.nan, raw=False)

cands["ret_kurt_20d"] = ret.apply(lambda c: roll_kurt(c, 20))

# alpha momentum vs equal-weight mkt
cands["alpha_mom_60d"] = (px / px.shift(60) - 1.0) - (mkt.rolling(60).mean() * 60)  # approx mkt 60d via daily mean
cands["mom_winrate_60d"] = (ret > 0).rolling(60).mean()

# semi-deviation ratio: upside vol / downside vol
up = ret.where(ret > 0, 0.0).rolling(60).std()
dn = ret.where(ret < 0, 0.0).rolling(60).std()
cands["semi_vol_ratio_60d"] = up / dn

# composite momentum z-score
def zscore(x, win=252):
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std()
    return (x - mu) / sd

m20 = px / px.shift(20) - 1.0
m60 = px / px.shift(60) - 1.0
m120 = px / px.shift(120) - 1.0
cands["composite_mom_z_20_60_120"] = (zscore(m20) + zscore(m60) + zscore(m120)) / 3.0

# MACD
ema12 = px.ewm(span=12, adjust=False).mean()
ema26 = px.ewm(span=26, adjust=False).mean()
cands["macd_12_26"] = (ema12 - ema26) / px

# short reversal
cands["ret_rev_5d"] = -(px / px.shift(5) - 1.0)

# vol-adj momentum variants
cands["vol_adj_mom_accel_10x40"] = vol_adj_mom_accel(fast=10, slow=40)
cands["vol_adj_mom_accel_30x90"] = vol_adj_mom_accel(fast=30, slow=90)

# WTI energy beta (commodity risk sensitivity)
cands["wti_beta_60d"] = beta_to(ret["WTI"])

# gold beta (safe-haven sensitivity)
cands["xau_beta_60d"] = beta_to(ret["XAU"])

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
          f"r252=({r['r_252'][0]:+.3f},{r['r_252'][1]:+.2f}) cov={coverage_assets(sig):.2f} to={turnover_rank(sig):.2f}{flag}")

# ---------- 3) library correlation for full-pass candidates ----------
print("\n=== LIBRARY CORRELATION (full-pass candidates vs library) ===")
lib = dict(existing)
lib.update({
    "mom_10d_skip5": px.shift(5) / px.shift(15) - 1.0,
    "mom_120d_skip5": px.shift(5) / px.shift(125) - 1.0,
    "vol_of_vol20x60": ret.rolling(20).std().rolling(60).std(),
})
vix_ret = macro_df["VIX"].pct_change()
vix_beta = beta_to(vix_ret)
lib["vix_beta_cond_60x20"] = -vix_beta * (macro_df["VIX"] / macro_df["VIX"].shift(20) - 1.0)

for name, (s, r, sig) in results.items():
    if abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840:
        best, key = 0.0, None
        for lname, lsig in lib.items():
            both = pd.concat([sig.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
            if len(both) < 30:
                continue
            rr = float(both["c"].corr(both["l"]))
            if abs(rr) > best:
                best, key = abs(rr), lname
        print(f"{name:26s} max_abs_lib_corr={best:.4f} (vs {key})")

print("\ndone")
