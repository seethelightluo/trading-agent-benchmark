"""miner_1 cycle: screen NEW factor families decorrelated from reversal library.
Focus: vol-regime, trend-quality, beta/co-movement, skewness, liquidity (volume),
and overnight/intraday splits. Each candidate gets IC1/ICIR/coverage/turnover/decay
plus max abs Spearman rho vs all 30 top-level library signal artifacts.
"""
import sys, os, json, glob, base64, gzip, pickle, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner1_common import (CUT, SYMBOLS, MACRO, load_close, build_returns,
                           factor_panel, ic_analysis, decay_analysis, coverage, turnover)

T0 = time.time()
closes = load_close()  # 15 tradable
macro = load_close(MACRO, dir_="../persistent/index_data")

print(f"[data] {len(SYMBOLS)} symbols, {len(MACRO)} macro, CUT={CUT.date()}", flush=True)
for s in SYMBOLS:
    df = closes[s]
    print(f"  {s:10s} {df.index.min().date()}..{df.index.max().date()} n={len(df)} "
          f"vol_nan={df['volume'].isna().mean():.2%} ohlc_nan={df[['open','high','low']].isna().mean().mean():.2%}", flush=True)

# ---------------- factor builders (per-symbol Series) ----------------
def rets(df, n=1):
    return np.log(df["close"]).diff(n)

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-12)

def vol_series(df, win=20):
    return rets(df, 1).rolling(win).std()

# 1) 20d realized vol z-score vs own trailing 120d history (vol regime)
def f_volz20(df, win=20, look=120):
    v = vol_series(df, win)
    m = v.rolling(look).mean()
    sd = v.rolling(look).std()
    return (v - m) / (sd + 1e-12)

# 2) short/long vol ratio
def f_volratio(df, short=10, long=60):
    return vol_series(df, short) / (vol_series(df, long) + 1e-12)

# 3) vol-of-vol: std of 20d vol over 60d / mean
def f_vov(df, vwin=20, look=60):
    v = vol_series(df, vwin)
    m = v.rolling(look).mean()
    return v.rolling(look).std() / (m + 1e-12)

# 4) downside/upside semi-vol ratio (vol asymmetry), 60d
def f_semivol(df, win=60):
    r = rets(df, 1)
    down = r.rolling(win).apply(lambda x: np.sqrt(np.mean(np.minimum(x, 0) ** 2)), raw=True)
    up = r.rolling(win).apply(lambda x: np.sqrt(np.mean(np.maximum(x, 0) ** 2)), raw=True)
    return down / (up + 1e-12)

# 5) R^2 of 60d linear trend (trend clarity, direction-agnostic)
def f_r2_trend(df, win=60):
    lp = np.log(df["close"])
    out = pd.Series(np.nan, index=lp.index)
    x = np.arange(win, dtype=float)
    for i in range(win - 1, len(lp)):
        y = lp.iloc[i - win + 1: i + 1].values
        if not np.all(np.isfinite(y)):
            continue
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        out.iloc[i] = 1 - ss_res / (ss_tot + 1e-12)
    return out

# 6) efficiency ratio 60d
def f_er(df, win=60):
    lp = np.log(df["close"])
    net = lp.diff(win).abs()
    path = lp.diff().abs().rolling(win).sum()
    return net / (path + 1e-12)

# 7) drawdown speed: max drawdown magnitude / days since high, 60d
def f_dd_speed(df, win=60):
    roll_high = df["close"].rolling(win).max()
    dd = df["close"] / roll_high - 1.0
    # days since rolling high
    since = np.arange(len(df)) - df["close"].rolling(win).apply(
        lambda x: np.nanargmax(x) if len(x) == win else np.nan, raw=True)
    speed = -dd / (since + 1.0)
    return speed

# 8) range expansion: (high-low)/close, 20d mean
def f_range_exp(df, win=20):
    rng = (df["high"] - df["low"]) / df["close"]
    return rng.rolling(win).mean()

# 9) 20d skewness of daily returns
def f_skew(df, win=20):
    return rets(df, 1).rolling(win).skew()

# 10) lag-1 autocorrelation of daily returns, 20d
def f_acorr(df, win=20):
    r = rets(df, 1)
    return r.rolling(win).apply(lambda x: pd.Series(x).autocorr(1) if np.std(x) > 0 else np.nan, raw=True)

# 11) risk-adjusted 60d momentum
def f_mom_ra(df, win=60):
    m = np.log(df["close"]).diff(win)
    return m / (vol_series(df, 20).rolling(win).mean() * np.sqrt(win) + 1e-12)

# 12) 5d reversal scaled by 20d vol (vol-adjusted reversal)
def f_rev5_voladj(df):
    r5 = np.log(df["close"]).diff(5)
    return -r5 / (vol_series(df, 20) + 1e-12)

# 13) rolling beta to SPX (equity beta), 60d
def f_beta_spx(df, win=60, mkt="SPX"):
    r = rets(df, 1)
    rm = rets(closes[mkt], 1).reindex(df.index)
    cov = r.rolling(win).cov(rm)
    var = rm.rolling(win).var()
    return cov / (var + 1e-12)

# 14) rolling beta to DXY, 60d (dollar sensitivity)
def f_beta_dxy(df, win=60):
    r = rets(df, 1)
    rm = rets(macro["DXY"], 1).reindex(df.index)
    cov = r.rolling(win).cov(rm)
    var = rm.rolling(win).var()
    return cov / (var + 1e-12)

# 15) rolling beta to VIX changes, 60d (risk-sensitivity)
def f_beta_vix(df, win=60):
    r = rets(df, 1)
    rm = macro["VIX"]["close"].reindex(df.index).diff()
    cov = r.rolling(win).cov(rm)
    var = rm.rolling(win).var()
    return cov / (var + 1e-12)

# 16) 20d cumulative overnight return (open_t / close_{t-1})
def f_overnight(df, win=20):
    on = np.log(df["open"] / df["close"].shift(1))
    return on.rolling(win).sum()

# 17) 20d cumulative intraday return (close_t / open_t)
def f_intraday(df, win=20):
    idr = np.log(df["close"] / df["open"])
    return idr.rolling(win).sum()

# 18) Amihud illiquidity 20d: mean(|ret|/volume), sign = liquidity
def f_amihud(df, win=20):
    r = rets(df, 1).abs()
    ill = r / (df["volume"] + 1e-9)
    return -ill.rolling(win).mean()

# 19) volume z-score: 20d volume vs 120d
def f_volz(df, vwin=20, look=120):
    v = df["volume"].rolling(vwin).mean()
    m = v.rolling(look).mean()
    sd = v.rolling(look).std()
    return (v - m) / (sd + 1e-12)

# 20) conditional: reversal only when own vol is above its median (mean-reversion regime)
def f_rev1_hi_vol(df, volwin=60):
    r1 = -rets(df, 1)
    v = vol_series(df, 20)
    vmed = v.rolling(volwin).median()
    return r1 * (v > vmed).astype(float)

# 21) momentum 60d skip 20 (mid-horizon trend)
def f_mom60_skip20(df):
    return np.log(df["close"]).diff(60).shift(20)

# 22) close location value multi-day: mean((close-low)/(high-low)) over 5d
def f_clv5(df):
    clv = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-12)
    return clv.rolling(5).mean()

# 23) intraday reversal 1d: -(close-open)/(high-low)
def f_intraday_rev1(df):
    return -(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)

cands = {
    "volz20_120": f_volz20,
    "volratio10_60": f_volratio,
    "volofvol60": f_vov,
    "semivol_ratio60": f_semivol,
    "r2_trend60": f_r2_trend,
    "er60": f_er,
    "dd_speed60": f_dd_speed,
    "range_exp20": f_range_exp,
    "skew20": f_skew,
    "acorr20": f_acorr,
    "mom_ra60": f_mom_ra,
    "rev5_voladj": f_rev5_voladj,
    "beta_spx60": f_beta_spx,
    "beta_dxy60": f_beta_dxy,
    "beta_vix60": f_beta_vix,
    "overnight20": f_overnight,
    "intraday20": f_intraday,
    "amihud20": f_amihud,
    "volz20": f_volz,
    "rev1_hi_vol": f_rev1_hi_vol,
    "mom60_skip20": f_mom60_skip20,
    "clv5": f_clv5,
    "intraday_rev1": f_intraday_rev1,
}

# ---------------- library signal artifacts ----------------
lib = {}
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f:
        continue
    try:
        d = json.load(open(f))
        sa = d.get("signal_artifact")
        if not sa:
            continue
        raw = base64.b64decode(sa["data_b64"])
        arr = np.frombuffer(gzip.decompress(raw), dtype=np.float32).reshape(sa["n_dates"], sa["n_symbols"])
        dates = pd.date_range(sa["date_start"], sa["date_end"], freq="B")
        syms = sa["symbols"]
        df = pd.DataFrame(arr, index=dates[: arr.shape[0]], columns=syms)
        lib[d["factor_id"] + "__" + f.split("/")[-1].replace(".json", "")] = df
    except Exception as e:
        print(f"[warn] skip {f}: {e}", flush=True)
print(f"[lib] loaded {len(lib)} signal artifacts", flush=True)

# ---------------- run candidates ----------------
results = {}
for name, fn in cands.items():
    t1 = time.time()
    try:
        panel = factor_panel(closes, fn)
        cov = coverage(panel, closes)
        to = turnover(panel)
        ic1 = ic_analysis(panel, closes, fwd_days=1)
        ic5 = ic_analysis(panel, closes, fwd_days=5)
        dec = decay_analysis(panel, closes)

        # max abs spearman rho vs library (flattened aligned pairs)
        maxrho, worst = 0.0, None
        for lid, ldf in lib.items():
            common_dates = panel.index.intersection(ldf.index)
            common_syms = [s for s in panel.columns if s in ldf.columns]
            a = panel.loc[common_dates, common_syms].values.ravel()
            b = ldf.loc[common_dates, common_syms].values.ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 500:
                continue
            rho = np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1]
            if np.isfinite(rho) and abs(rho) > maxrho:
                maxrho, worst = abs(rho), lid
        results[name] = {"cov": cov, "turnover": to, "ic1": ic1, "ic5": ic5,
                         "decay": dec, "max_rho": float(maxrho), "worst": worst}
        print(f"{name:16s} cov={cov:.3f} turn={to:.3f} | "
              f"IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.3f} n={ic1['n_dates']} | "
              f"IC5={ic5['ic']:+.4f} | maxrho={maxrho:.3f} vs {str(worst)[:34]} ({time.time()-t1:.0f}s)", flush=True)
    except Exception as e:
        print(f"{name:16s} ERROR {e}", flush=True)

print(f"\nelapsed {time.time()-T0:.0f}s", flush=True)

print("\n--- ADMISSION GATE (|IC|>=0.007, |ICIR|>=0.084, cov>=0.5, maxrho<0.5) ---")
passers = {}
for name, r in results.items():
    ic, icir = r["ic1"]["ic"], r["ic1"]["icir"]
    if (abs(ic) >= 0.007 and abs(icir) >= 0.084 and r["cov"] >= 0.5 and r["max_rho"] < 0.5):
        passers[name] = r
        print(f"  PASS {name}: IC={ic:+.4f} ICIR={icir:+.3f} cov={r['cov']:.3f} turn={r['turnover']:.3f} "
              f"maxrho={r['max_rho']:.3f} decay1={r['decay'].get(1):.4f} decay5={r['decay'].get(5):.4f}")
    else:
        print(f"  fail {name}: IC={ic:+.4f} ICIR={icir:+.3f} cov={r['cov']:.3f} maxrho={r['max_rho']:.3f}")

with open("scripts/miner1_macro_voltrend_results.json", "w") as fh:
    json.dump({k: {kk: (vv if not isinstance(vv, dict) else
                         {kkk: (round(vvv, 4) if isinstance(vvv, float) else vvv) for kkk, vvv in vv.items()})
                   for kk, vv in v.items()} for k, v in results.items()}, fh, indent=1, default=str)
print("saved scripts/miner1_macro_voltrend_results.json")
