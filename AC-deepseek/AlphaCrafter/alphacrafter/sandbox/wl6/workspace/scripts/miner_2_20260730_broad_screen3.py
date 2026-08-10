import sys, math, warnings
sys.path.insert(0, "scripts")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from factor_validation_lib import TRADABLE, load_panel, load_macro, align_fwd_returns, rank_ic_series, ic_analysis, print_report

MAX_D = "2026-07-29"
panel = load_panel(max_date=MAX_D)
ret = panel.pct_change()

# ---------- OHLC panels ----------
def load_ohlc(max_date=MAX_D):
    h, l, v = {}, {}, {}
    for sym in TRADABLE:
        try:
            df = get_stock_daily_data(symbol=sym, days=4000)
        except Exception:
            df = None
        if df is not None and len(df) > 100:
            idx = pd.to_datetime(df["date"])
            if max_date is not None:
                m = idx <= pd.Timestamp(max_date)
                df = df[m]; idx = idx[m]
            h[sym] = pd.Series(df["high"].astype(float).values, index=idx)
            l[sym] = pd.Series(df["low"].astype(float).values, index=idx)
            v[sym] = pd.Series(df["volume"].astype(float).values, index=idx)
    return (pd.DataFrame(h).sort_index(), pd.DataFrame(l).sort_index(), pd.DataFrame(v).sort_index())

hi, lo, vol = load_ohlc()
print("OHLC panels:", hi.shape, vol.shape)

# ---------- macro ----------
dxy = load_macro("DXY", max_date=MAX_D)
vix = load_macro("VIX", max_date=MAX_D)
usdjpy = load_macro("USDJPY", max_date=MAX_D)
usdcny = load_macro("USDCNY", max_date=MAX_D)
eurusd = load_macro("EURUSD", max_date=MAX_D)

def reindex_macro(s):
    return s.reindex(panel.index)

dxy_r, vix_r = reindex_macro(dxy), reindex_macro(vix)
usdjpy_r, usdcny_r = reindex_macro(usdcny), reindex_macro(usdjpy)

# ---------- factor builders ----------
C = {}
def add(name, f):
    if f is not None:
        C[name] = f.reindex(panel.index)

sma = panel.rolling(20).mean(); sma60 = panel.rolling(60).mean()
std20 = ret.rolling(20).std(); std60 = ret.rolling(60).std()
min20 = panel.rolling(20).min(); max20 = panel.rolling(20).max()
max60 = panel.rolling(60).max()
up = ret.clip(lower=0); dn = ret.clip(upper=0)

# momentum / trend
add("mom20s5", panel.shift(5)/panel.shift(25) - 1.0)
add("mom60s10", panel.shift(10)/panel.shift(70) - 1.0)
add("tsmom_sma20", panel/sma - 1.0)
add("tsmom_sma60", panel/sma60 - 1.0)
add("eff_ratio20", (panel - panel.shift(20)).abs() / ret.abs().rolling(20).sum())
add("boll_z20", (panel - sma)/std20)
add("stoch_20", (panel - min20)/(max20 - min20))
rsi = 100.0 - 100.0/(1.0 + ret.clip(lower=0).rolling(14).mean()/ret.clip(upper=0).abs().rolling(14).mean())
add("rsi14", rsi)

# volatility / risk
dvol = np.sqrt((dn**2).rolling(20).mean()) * np.sqrt(252)
add("dvol20", dvol)
add("vol_ratio10_60", ret.rolling(10).std()/ret.rolling(60).std())
add("skew20", ret.rolling(20).skew())
add("maxdd60", panel/max60 - 1.0)
rng = (hi - lo)/panel
add("range20", rng.rolling(20).mean())
upvol = up.rolling(20).std(); dnvol = dn.rolling(20).std()
add("updown_vol20", upvol/dnvol)

# cross-asset / macro conditional
def roll_beta(y, x, w=60):
    xr = x.pct_change()
    yr = y.pct_change()
    cov = yr.rolling(w).cov(xr)
    var = xr.rolling(w).var()
    return cov/var

add("beta_spx60", roll_beta(panel, panel["SPX"], 60))
add("beta_vix60", roll_beta(panel, vix_r, 60))
add("beta_dxy60", roll_beta(panel, dxy_r, 60))
add("beta_ndx60", roll_beta(panel, panel["NDX"], 60))
vix_med60 = vix_r.rolling(60).median()
dxy_med60 = dxy_r.rolling(60).median()
mom20 = panel.shift(5)/panel.shift(25) - 1.0
add("vix_cond_mom", mom20 * np.where(vix_r <= vix_med60, 1.0, -1.0))
add("dxy_cond_mom", mom20 * np.where(dxy_r <= dxy_med60, 1.0, -1.0))
add("jpy_carry_cond", mom20 * np.where(usdjpy_r > usdjpy_r.rolling(60).mean(), 1.0, -1.0))

# volume (9 assets only)
v5 = vol.rolling(5).mean(); v60 = vol.rolling(60).mean(); v20m = vol.rolling(20).mean(); v20s = vol.rolling(20).std()
add("vol_ratio_5_60", v5/v60)
add("volume_z20", (vol - v20m)/v20s)
add("amihud20", (ret.abs()/vol).rolling(20).mean())

# combos
add("risk_adj_mom20", mom20/std20)
add("mom60_over_vol60", (panel.shift(10)/panel.shift(70) - 1.0)/std60)
add("dd_recovery20", panel/panel.rolling(20).min() - 1.0)

# ---------- screen ----------
fwd10 = align_fwd_returns(panel, 10)
print(f"{'factor':<22}{'ic':>8}{'icir':>8}{'hit':>7}{'n':>6}{'cov':>7}  gate | sub-IC 20-22 / 23-26")
rows = []
for name, f in C.items():
    ics = rank_ic_series(f, fwd10).dropna()
    if len(ics) == 0:
        continue
    ic = float(ics.mean()); icir = ic/(float(ics.std(ddof=1)) or np.nan)
    hit = float((ics > 0).mean())
    cov = float(f.notna().mean().mean())
    flag = "PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else ""
    s1 = ics[(ics.index >= "2020-01-01") & (ics.index <= "2022-12-31")].mean() if len(ics) else np.nan
    s2 = ics[(ics.index >= "2023-01-01")].mean()
    print(f"{name:<22}{ic:>8.4f}{icir:>8.4f}{hit:>7.3f}{len(ics):>6d}{cov:>7.2f}  {flag:<5} {s1:.4f} / {s2:.4f}")
    rows.append((name, abs(ic), abs(icir), ic, icir, hit, len(ics), cov))

print("\n--- ranked by |IC|*|ICIR| ---")
for name, aic, aicir, ic, icir, hit, n, cov in sorted(rows, key=lambda r: r[1]*r[2], reverse=True):
    flag = "PASS" if (aic >= 0.007 and aicir >= 0.084) else ""
    print(f"{name:<22}{ic:>8.4f}{icir:>8.4f}{flag}")
