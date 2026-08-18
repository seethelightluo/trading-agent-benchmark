"""
miner_2 batch screen 2027-10-07 cycle (data visible through 2027-10-06).

Context: live ensemble beta_vix_60d_neg(0.36)/mom_120d_skip5(0.30)/vol_of_vol20x60(0.18)/
low_vol_20d(0.16,dir=-1). Last block 20270923-20271007 +2.86%, first block above 1M net,
regime bull (trend 1.45). Prior 2027-09-23 screen: no new factor passed full-window gate;
strong recent performers were dd_speed_60d (2027+ ICIR 0.28) and vix_prem_beta_60d
(full IC -0.0606 ICIR -0.158 but librho=0.500 exactly, driven by beta_vix_60d_neg rho=-0.50),
gain_loss_asym_20d (librho 0.584 vs down_vol_ratio_20x120).

Goal:
 1) Re-validate the 8 library factors (drift check full / online / 2027+ / last-250d).
 2) Discover NEW orthogonal factors passing |IC|>=0.0070 & |ICIR|>=0.0840 at H=10 on the
    15-instrument universe (>=250 IC dates, >=8 valid instruments/date, max abs library
    correlation < 0.5); PERSIST gate-passers with signal artifacts (base64:zlib:csv).
 3) New candidate families:
    A) vix_prem_beta_60d_orth1  : vix premium beta residualized on beta_vix_60d_neg (rolling 250d OLS)
    B) gain_loss_asym_20d_orth2 : gain/loss asymmetry residualized on down_vol_ratio_20x120 + mom_10d_skip5
    C) dd_speed_60d / dd_speed_120d (re-test recent strength)
    D) down_beta_ew_60d_neg     : downside beta vs EW market (beta on down-market days), negated
    E) gap_share_20d            : overnight (gap) vol share of total 20d vol
    F) mom_250d_skip5           : long-horizon momentum
    G) skew_120d_neg            : negative 120d skewness
    H) risk_adj_mom_60d         : mom60 / vol20
    I) corr_ew_60d_neg          : negative 60d correlation to EW portfolio (diversifier)
    J) cvar_20d_neg             : negative 20d CVaR (5% worst)
    K) vol_jump_5d              : vol20/vol20.shift(5)-1 (recent vol expansion)
    L) park_vol_20d_neg         : negative Parkinson 20d vol (high-low based)
    M) up_down_60d              : 60d up-move magnitude / down-move magnitude
    N) ret_consistency_20d      : fraction of positive days over 20d
    O) beta_spx_cond_60x20      : beta to SPX * sign(SPX 20d return)
    P) vol_of_vol10x40          : short-window vol-of-vol
    Q) mom_vix_regime_120       : mom120 * sign(VIX 20d change) (risk-on regime momentum)
    R) vol_adj_dd_speed_60d     : dd_speed_60d / vol20 (risk-adjusted drawdown speed)

Pure research; no account/date mutation, no backtest/step.
"""
import json, time, hashlib, base64, zlib
import numpy as np
import pandas as pd

VISIBLE = "2027-10-06"
H_ADMIT = 10
MIN_IC_DATES = 250
MIN_INSTR = 8
IC_TH, ICIR_TH = 0.0070, 0.0840
CORR_TH = 0.5
WARM_END = pd.Timestamp("2026-07-15")
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

t0 = time.time()


def load_close(sym, cutoff, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date").sort_index()


def load_panel(cutoff):
    closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
    for s in TRADABLE:
        df = load_close(s, cutoff)
        closes[s] = df["close"].astype(float)
        vols[s] = df["volume"].astype(float) if "volume" in df else pd.Series(np.nan, index=df.index)
        highs[s] = df["high"].astype(float) if "high" in df else pd.Series(np.nan, index=df.index)
        lows[s] = df["low"].astype(float) if "low" in df else pd.Series(np.nan, index=df.index)
        opens[s] = df["open"].astype(float) if "open" in df else pd.Series(np.nan, index=df.index)
    px = pd.DataFrame(closes).dropna(how="all")
    return px, pd.DataFrame(vols), pd.DataFrame(highs), pd.DataFrame(lows), pd.DataFrame(opens)


px, vol, hi, lo, op = load_panel(VISIBLE)
ret = px.pct_change()
print(f"panel: {px.shape} {px.index.min().date()}..{px.index.max().date()} ({time.time()-t0:.1f}s)", flush=True)

obs = {s: load_close(s, VISIBLE, INDEX_DIR)["close"].astype(float) for s in OBS}
vix = obs["VIX"]; vixr = vix.pct_change(); vix_move20 = (vix / vix.shift(20) - 1.0)
us10y = px["US10Y"]; cn10y = px["CN10Y"]
us10y_r = us10y.pct_change()
us10y_m20 = us10y / us10y.shift(20) - 1.0


def mp(w, frac=2):
    return min(max(5, w // (frac or 1)), w)


def rs(x, w):
    return x.rolling(w, min_periods=mp(w)).std()


def rm(x, w):
    return x.rolling(w, min_periods=mp(w)).mean()


def beta_of(a, m, w):
    m = m.reindex(a.index)
    mdf = pd.DataFrame({c: m for c in a.columns}, index=a.index)
    cov = a.rolling(w, min_periods=mp(w, 2)).cov(mdf)
    var = mdf.rolling(w, min_periods=mp(w, 2)).var().replace(0, np.nan)
    return cov / var


def rolling_ols_resid(y, Xlist, w=250, min_w=60):
    """Per-column rolling OLS residual of y on [1]+Xlist (each X a DataFrame, same asset cols)."""
    Xlist = [x.reindex(y.index) for x in Xlist]
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    idx = y.index
    for c in y.columns:
        yv = y[c].values.astype(float)
        Xcols = [x[c].values.astype(float) for x in Xlist]
        Xv = np.column_stack(Xcols) if Xcols else np.empty((len(idx), 0))
        bad = np.isnan(yv) | np.isnan(Xv).any(axis=1)
        yc = np.where(bad, np.nan, yv)
        Xc = np.where(bad[:, None], np.nan, Xv)
        res = np.full(len(idx), np.nan)
        for t in range(w, len(idx)):
            sl = slice(t - w, t)
            ys = yc[sl]
            Xs = Xc[sl]
            m = ~(np.isnan(ys) | np.isnan(Xs).any(axis=1))
            if m.sum() < min_w:
                continue
            A = np.column_stack([np.ones(m.sum()), Xs[m]])
            b = ys[m]
            try:
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                pred = coef[0] + Xc[t] @ coef[1:]
                res[t] = yc[t] - pred
            except Exception:
                continue
        out[c] = res
    return out


# ---------------- library signals (8 persisted factors, recomputed) ----------------
lib = {}
lib["mom_10d_skip5"] = (px.shift(5) / px.shift(15) - 1.0)
lib["mom_120d_skip5"] = (px.shift(5) / px.shift(125) - 1.0)
lib["vol_of_vol20x60"] = rs(ret, 20).rolling(60, min_periods=mp(60)).std()
lib["vix_beta_cond_60x20"] = (-beta_of(ret, vixr, 60)).mul(vix_move20.reindex(ret.index), axis=0)
lib["beta_vix_60d_neg"] = -beta_of(ret, vixr, 60)
lib["low_vol_20d"] = -rs(ret, 20)
down = (ret.clip(upper=0) * -1.0)
lib["down_vol_ratio_20x120"] = -(rs(down, 20) / rs(down, 120).replace(0, np.nan))
lib["beta_cn10y_60d"] = beta_of(ret, cn10y.pct_change(), 60)

# ---------------- new candidates ----------------
C = {}
vol5 = rs(ret, 5); vol10 = rs(ret, 10); vol20 = rs(ret, 20)
vol60 = rs(ret, 60); vol120 = rs(ret, 120)
ret5 = px.pct_change(5); ret10 = px.pct_change(10); ret20 = px.pct_change(20)
ret60 = px.pct_change(60); ret120 = px.pct_change(120); ret250 = px.pct_change(250)

# A) vix premium beta residualized on beta_vix_60d_neg
spx_rvol20 = rs(px["SPX"].pct_change(), 20)
prem = (vix / spx_rvol20.reindex(vix.index).replace(0, np.nan)).diff()
vix_prem_beta = beta_of(ret, prem, 60)
C["vix_prem_beta_60d"] = vix_prem_beta
print("residualizing vix_prem_beta (rolling 250d OLS on beta_vix_60d_neg)...", flush=True)
C["vix_prem_beta_60d_orth1"] = rolling_ols_resid(vix_prem_beta, [lib["beta_vix_60d_neg"]])

# B) gain/loss asymmetry residualized on down_vol_ratio_20x120 + mom_10d_skip5
up = ret.clip(lower=0); dn = (ret.clip(upper=0) * -1.0)
gla = rs(up, 20) / rs(dn, 20).replace(0, np.nan)
C["gain_loss_asym_20d"] = gla
C["gain_loss_asym_20d_orth2"] = rolling_ols_resid(gla, [lib["down_vol_ratio_20x120"], lib["mom_10d_skip5"]])

# C) drawdown speed 60d / 120d
def dd_speed(px, w):
    hh = px.rolling(w, min_periods=mp(w)).max()
    dd = px / hh - 1.0
    days_since_high = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for c in px.columns:
        ch = px[c]
        run = pd.Series(np.nan, index=ch.index)
        cur = np.nan
        for i in range(len(ch)):
            v = ch.iloc[i]
            if np.isnan(v):
                cur = np.nan
            else:
                cur = 0 if (np.isnan(cur) or v >= hh[c].iloc[i] * 0.9999) else cur + 1
            run.iloc[i] = cur
        days_since_high[c] = run
    return (dd / days_since_high.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

print("building dd_speed...", flush=True)
C["dd_speed_60d"] = dd_speed(px, 60)
C["dd_speed_120d"] = dd_speed(px, 120)

# D) downside beta vs EW market (beta on down-market days), negated
ew = ret.mean(axis=1)
down_day = ew < 0
ret_neg = ret.where(down_day, np.nan)
ew_neg = ew.where(down_day, np.nan)
C["down_beta_ew_60d_neg"] = -beta_of(ret_neg, ew_neg, 60)

# E) overnight (gap) vol share of total 20d vol
gap = (op / px.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
intra = (px / op - 1.0).replace([np.inf, -np.inf], np.nan)
C["gap_share_20d"] = rs(gap, 20) / (rs(gap, 20) + rs(intra, 20)).replace(0, np.nan)

# F) long-horizon momentum
C["mom_250d_skip5"] = px.shift(5) / px.shift(255) - 1.0

# G) negative 120d skewness
C["skew_120d_neg"] = -ret.rolling(120, min_periods=mp(120)).skew()

# H) risk-adjusted momentum 60d
C["risk_adj_mom_60d"] = ret60 / vol20.replace(0, np.nan)

# I) negative correlation to EW portfolio
C["corr_ew_60d_neg"] = -ret.rolling(60, min_periods=mp(60, 2)).corr(ew)

# J) negative 20d CVaR (mean of 5% worst days)
def cvar_neg(r, w, q=0.05):
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for c in r.columns:
        rv = r[c].values.astype(float)
        res = np.full(len(rv), np.nan)
        for t in range(w, len(rv)):
            win = rv[t - w:t]
            win = win[~np.isnan(win)]
            if len(win) < mp(w):
                continue
            k = max(1, int(np.ceil(q * len(win))))
            worst = np.sort(win)[:k]
            res[t] = -worst.mean()
        out[c] = res
    return out

print("building cvar_20d...", flush=True)
C["cvar_20d_neg"] = cvar_neg(ret, 20)

# K) vol jump 5d
C["vol_jump_5d"] = vol20 / vol20.shift(5) - 1.0

# L) negative Parkinson 20d vol
hl = np.log(hi / lo).replace([np.inf, -np.inf], np.nan)
C["park_vol_20d_neg"] = -np.sqrt(hl.rolling(20, min_periods=mp(20)).mean() / (4 * np.log(2)))

# M) up/down magnitude ratio 60d
up60 = ret.clip(lower=0).rolling(60, min_periods=mp(60)).sum()
dn60 = (ret.clip(upper=0) * -1.0).rolling(60, min_periods=mp(60)).sum()
C["up_down_60d"] = up60 / dn60.replace(0, np.nan)

# N) return consistency: fraction of positive days over 20d
C["ret_consistency_20d"] = (ret > 0).rolling(20, min_periods=mp(20)).mean()

# O) conditional market beta: beta to SPX * sign(SPX 20d ret)
bspx = beta_of(ret, px["SPX"].pct_change(), 60)
spx_ret20 = px["SPX"].pct_change(20)
C["beta_spx_cond_60x20"] = bspx.mul(spx_ret20.reindex(ret.index).apply(np.sign), axis=0)

# P) short-window vol-of-vol
C["vol_of_vol10x40"] = rs(ret, 10).rolling(40, min_periods=mp(40)).std()

# Q) momentum conditioned on VIX falling regime
C["mom_vix_regime_120"] = lib["mom_120d_skip5"].mul(vix_move20.reindex(ret.index).apply(np.sign), axis=0)

# R) risk-adjusted drawdown speed
C["vol_adj_dd_speed_60d"] = C["dd_speed_60d"] / vol20.replace(0, np.nan)

print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)


def fast_ic_series(factor, fwd, min_valid=MIN_INSTR):
    common = factor.index.intersection(fwd.index)
    fr = factor.reindex(common).rank(axis=1, pct=True)
    rr = fwd.reindex(common).rank(axis=1, pct=True)
    mask = fr.isna().values | rr.isna().values
    nvalid = (~mask).sum(axis=1)
    F = np.ma.array(fr.values, mask=mask)
    R = np.ma.array(rr.values, mask=mask)
    Fm = F - F.mean(axis=1, keepdims=True)
    Rm = R - R.mean(axis=1, keepdims=True)
    num = (Fm * Rm).sum(axis=1)
    den = np.sqrt((Fm ** 2).sum(axis=1) * (Rm ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic = np.ma.filled(ic, np.nan)
    ic[nvalid < min_valid] = np.nan
    return pd.Series(ic, index=common)


def ic_summary(ic):
    ic = ic.dropna()
    if len(ic) < 30:
        return np.nan, np.nan, np.nan, len(ic)
    m = float(ic.mean())
    s = float(ic.std(ddof=1))
    icir = m / s if s > 0 else 0.0
    hit = float((ic > 0).mean())
    return m, icir, hit, len(ic)


def turnover_10d(f):
    rk = f.rank(axis=1, pct=True)
    chg = rk.diff(10).abs().mean(axis=1).mean()
    return float(chg) if np.isfinite(chg) else np.nan


def coverage(f):
    fv = f.notna()
    cov_asset_days = float(fv.values.mean()) if fv.size else np.nan
    n_valid = fv.sum(axis=1)
    cov_dates_ge8 = float((n_valid >= 8).mean())
    return cov_asset_days, cov_dates_ge8


def max_lib_corr(f, libs):
    best, det = 0.0, {}
    fs = f.stack().rename("c")
    for k, sig in libs.items():
        both = pd.concat([fs, sig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rho = float(both["c"].rank().corr(both["l"].rank()))
        det[k] = round(rho, 3)
        best = max(best, abs(rho))
    return best, det


fwd10 = px.shift(-H_ADMIT) / px - 1.0
fwd_all = {h: px.shift(-h) / px - 1.0 for h in (1, 2, 3, 5, 10, 20)}
sub_windows = {"full": None, "warm": WARM_END, "2024+": pd.Timestamp("2024-01-01"),
               "2025+": pd.Timestamp("2025-01-01"), "2026+": pd.Timestamp("2026-01-01"),
               "online": pd.Timestamp("2026-07-16"), "2027+": pd.Timestamp("2027-01-01"),
               "last250": pd.Timestamp(px.index[-1]) - pd.Timedelta(days=365)}

results = {}
print(f"\n{'name':<26}{'IC':>8s}{'ICIR':>8s}{'hit':>6s}{'n':>6s}  {'librho':>7s}  {'turn':>6s}  "
      f"{'2027+IC':>8s}{'2027+IR':>8s} {'onlineIC':>9s}{'onlineIR':>9s}  {'decay10/20':>11s}", flush=True)
for name, f in {**C, **lib}.items():
    f = f.reindex(px.index)
    ic = fast_ic_series(f, fwd10)
    m, icir, hit, n = ic_summary(ic)
    lc, det = max_lib_corr(f, lib)
    turn = turnover_10d(f)
    ca, cd = coverage(f)
    rec = {}
    for wname, wstart in sub_windows.items():
        icw = ic if wname == "full" else ic[ic.index >= wstart]
        mm, ii, _, nn = ic_summary(icw)
        rec[wname] = (round(mm, 4), round(ii, 4)) if nn > 50 else None
    dec = {}
    for h, fh in fwd_all.items():
        ich = fast_ic_series(f, fh)
        mm, ii, _, _ = ic_summary(ich)
        dec[h] = (round(mm, 4), round(ii, 4)) if np.isfinite(mm) else None
    results[name] = {"ic": m, "icir": icir, "hit": hit, "n": n, "librho": lc,
                     "turn": turn, "cov_asset_days": ca, "cov_dates_ge8": cd,
                     "sub": rec, "decay": dec, "det": det}
    d10 = dec.get(10, (None, None))[0]
    d20 = dec.get(20, (None, None))[0]
    s27 = rec.get("2027+", (None, None))
    son = rec.get("online", (None, None))
    print(f"{name:<26}{m:>8.4f}{icir:>8.3f}{hit:>6.2f}{n:>6d}  {lc:>7.3f}  {turn:>6.2f}  "
          f"{s27[0] if s27 else float('nan'):>8.4f}{s27[1] if s27 else float('nan'):>8.3f} "
          f"{son[0] if son else float('nan'):>9.4f}{son[1] if son else float('nan'):>9.3f}  "
          f"{d10:>6.4f}/{d20:>6.4f}", flush=True)

print(f"\n--- gate check (|IC|>={IC_TH}, |ICIR|>={ICIR_TH}, n>={MIN_IC_DATES}, librho<{CORR_TH}) ---", flush=True)
passers = []
for name, r in results.items():
    if abs(r["ic"]) >= IC_TH and abs(r["icir"]) >= ICIR_TH and r["n"] >= MIN_IC_DATES and r["librho"] < CORR_TH:
        passers.append(name)
        print(f"PASS {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f} "
              f"turn={r['turn']:.2f} cov={r['cov_asset_days']:.3f}/{r['cov_dates_ge8']:.3f} sub={r['sub']}", flush=True)
    else:
        print(f"fail {name}: IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} librho={r['librho']:.3f}", flush=True)

print(f"\n--- library drift flags (2027+/online/last250 |IC|<{IC_TH} or sign flip) ---", flush=True)
for name in lib:
    r = results[name]
    flag = []
    for wname in ("2027+", "online", "last250"):
        s = r["sub"].get(wname)
        if s and (abs(s[0]) < IC_TH or (s[0] * r["ic"] < 0)):
            flag.append(f"{wname} IC={s[0]:.4f} ICIR={s[1]:.3f}")
    if r["librho"] >= CORR_TH:
        flag.append(f"librho={r['librho']:.3f}")
    print(f"{name}: full IC={r['ic']:.4f} ICIR={r['icir']:.3f} n={r['n']} -> {'FLAG ' + '; '.join(flag) if flag else 'ok'}", flush=True)

with open("scripts/miner_2_20271007_screen_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print(f"\nTOTAL TIME {time.time()-t0:.1f}s passers={passers} (results saved)", flush=True)
