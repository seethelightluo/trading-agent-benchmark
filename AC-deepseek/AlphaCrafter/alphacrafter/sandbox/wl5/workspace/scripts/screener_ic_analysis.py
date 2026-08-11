"""Screener: fresh cross-sectional IC analysis (data <= 2026-12-30 only).

Vectorized rolling factor computation. Read-only; no account/date writes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

CUTOFF = "2026-12-30"
DATA = Path("../persistent/stock_data")
IDX = Path("../persistent/index_data")
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_close(path, cutoff):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff)]
    return df.set_index("date")["close"].astype(float)


def build_panel():
    closes = {}
    for a in ASSETS:
        try:
            s = load_close(DATA / f"{a}.csv", CUTOFF)
        except Exception:
            s = load_close(IDX / f"{a}.csv", CUTOFF)
        closes[a] = s
    return pd.DataFrame(closes).sort_index()


def rank_ic(panel, fvals, horizon=10, min_valid=8):
    fwd = panel.shift(-horizon) / panel - 1.0
    common = fvals.index.intersection(fwd.index)
    f, r = fvals.loc[common], fwd.loc[common]
    ics = []
    for dt in common:
        x, y = f.loc[dt].astype(float), r.loc[dt].astype(float)
        m = x.notna() & y.notna()
        if m.sum() >= min_valid:
            ics.append((dt, np.corrcoef(x[m].rank(), y[m].rank())[0, 1]))
    return pd.DataFrame(ics, columns=["date", "ic"]).set_index("date")


def ic_stats(ics, n):
    if ics is None or len(ics) == 0:
        return None
    tail = ics.iloc[-n:]
    ic = float(tail["ic"].mean())
    sd = float(tail["ic"].std(ddof=1))
    icir = ic / sd * np.sqrt(252 / 10) if sd > 0 else 0.0
    return {"ic": round(ic, 4), "icir": round(icir, 3), "hit": round(float((tail["ic"] > 0).mean()), 3), "n": len(tail)}


def main():
    panel = build_panel()
    rets = panel.pct_change()
    mkt = panel.mean(axis=1)
    mkt_r = mkt.pct_change()
    dxy_r = load_close(IDX / "DXY.csv", CUTOFF).pct_change()
    vix = load_close(IDX / "VIX.csv", CUTOFF)
    vix_r = vix.pct_change()
    wti_r = panel["WTI"].pct_change()

    print(f"Panel: {panel.index.min().date()} .. {panel.index.max().date()} n={len(panel)}")

    trend20 = float(mkt.tail(20).mean())
    ma60 = float(mkt.tail(60).mean())
    avg_px = float(mkt.iloc[-1])
    rv20 = float(mkt_r.tail(20).std() * np.sqrt(252))
    rv60 = float(mkt_r.tail(60).std() * np.sqrt(252))
    mom60 = float(mkt.iloc[-1] / mkt.iloc[-61] - 1.0)
    mom120 = float(mkt.iloc[-1] / mkt.iloc[-121] - 1.0)
    disp20 = float(rets.tail(20).std(axis=1).mean())
    corr = rets.tail(60).corr()
    avg_corr = float((corr.values[np.triu_indices_from(corr.values, 1)]).mean())
    print("--- REGIME ---")
    print(f"avg_px={avg_px:.0f} ma60={ma60:.0f} trend20={trend20:.5f} bearish_gate={trend20<0 and avg_px<ma60}")
    print(f"mkt mom60={mom60:.3f} mom120={mom120:.3f} rv20={rv20:.3f} rv60={rv60:.3f} disp20={disp20:.4f} avg_corr60={avg_corr:.3f}")
    print(f"VIX={vix.iloc[-1]:.1f} VIX20d_chg={vix.iloc[-1]/vix.iloc[-21]-1:.3f} DXY={load_close(IDX/'DXY.csv',CUTOFF).iloc[-1]:.2f}")

    print("\n--- per-asset r60 / vol20 ---")
    for a in ASSETS:
        r60 = float(panel[a].iloc[-1] / panel[a].iloc[-61] - 1.0)
        v20 = float(rets[a].tail(20).std() * np.sqrt(252))
        print(f"  {a:10s} r60={r60:7.3f}  vol20={v20:5.3f}  last={panel[a].iloc[-1]:.1f}")

    print("\n--- vectorized factor series ---")
    F = {}

    # trend_r2_30_signed: rolling OLS R2 of log price on t, signed by slope
    lp = np.log(panel)
    x = np.arange(len(panel))
    def roll_ols_r2(ser):
        y = ser.rolling(30, min_periods=18).apply(
            lambda w: (lambda yy, xx: (np.cov(yy, xx)[0, 1] ** 2 / (np.var(yy) * np.var(xx)),
                                       np.copysign(1.0, np.cov(yy, xx)[0, 1])))(w, x[: len(w)]), raw=True)
        return y
    # simpler: covariance of y with t / (std_y * std_t), squared, signed
    ys = lp
    xs = pd.Series(x, index=panel.index)
    def _signed_r2(w):
        if len(w) < 18:
            return np.nan
        cov = float(np.cov(w, np.arange(len(w)))[0, 1])
        vy = float(np.var(w))
        if vy <= 0:
            return np.nan
        return np.copysign(cov * cov / (vy * np.var(np.arange(len(w)))), cov)
    F["trend_r2_30_signed"] = ys.rolling(30, min_periods=18).apply(_signed_r2, raw=True)

    # semi_down_ratio_20
    down = (rets.clip(upper=0) ** 2).rolling(20, min_periods=10).mean() ** 0.5
    up = (rets.clip(lower=0) ** 2).rolling(20, min_periods=10).mean() ** 0.5
    F["semi_down_ratio_20"] = down / up.replace(0, np.nan) - 1.0

    # mom_120d_skip5
    F["mom_120d_skip5"] = panel.shift(6) / panel.shift(126) - 1.0
    # mom_10d_skip5
    F["mom_10d_skip5"] = panel.shift(6) / panel.shift(16) - 1.0

    # vol_of_vol20x60
    v20 = rets.rolling(20).std()
    F["vol_of_vol20x60"] = v20.rolling(60).std()

    # kurt_20
    F["kurt_20"] = rets.rolling(20, min_periods=8).kurt()

    # time_under_water_120: days since last 120d rolling high
    def _days_since_high(w):
        w = np.asarray(w, dtype=float)
        roll = np.maximum.accumulate(w)
        mask = w == roll
        idx = np.flatnonzero(mask)
        return float(len(w) - 1 - idx[-1]) if len(idx) else float(len(w))
    F["time_under_water_120"] = panel.rolling(120, min_periods=60).apply(_days_since_high, raw=True)

    # betas: rolling cov(a,m)/var(m), 60d
    def _beta(asset_panel, mkt_r):
        a = asset_panel
        m = mkt_r
        both = pd.concat([a, m], axis=1)
        cov = both.rolling(60, min_periods=30).cov().iloc[0::2, 1].unstack() if False else None
        # manual
        am = pd.concat([a, m], axis=1)
        cov_a_m = a.rolling(60, min_periods=30).cov(m)
        var_m = m.rolling(60, min_periods=30).var()
        return cov_a_m / var_m.replace(0, np.nan)

    F["dxy_beta_60"] = _beta(rets, dxy_r)
    F["WTI_BETA_60"] = _beta(rets, wti_r)

    # vix_beta_cond_60x20 = -beta(asset, vix_ret,60) * (vix/vix.shift(20)-1)
    vix_beta = _beta(rets, vix_r)
    vix_chg = vix / vix.shift(20) - 1.0
    F["vix_beta_cond_60x20"] = -vix_beta.mul(vix_chg, axis=0)

    for fid, fv in F.items():
        print(f"  {fid:22s} last coverage {fv.iloc[-1].notna().sum()}/15")

    dirs = {"trend_r2_30_signed": 1, "semi_down_ratio_20": -1, "mom_120d_skip5": 1,
            "mom_10d_skip5": 1, "vol_of_vol20x60": 1, "kurt_20": 1,
            "time_under_water_120": -1, "dxy_beta_60": 1, "WTI_BETA_60": 1,
            "vix_beta_cond_60x20": -1}

    print("\n--- rank IC vs 10d fwd returns (raw; sign vs dir) ---")
    for fid, fv in F.items():
        ics = rank_ic(panel, fv)
        s_full, s120, s60 = ic_stats(ics, len(ics)), ic_stats(ics, 120), ic_stats(ics, 60)
        d = dirs[fid]
        print(f"{fid:22s} dir={d:+d} full_ic={s_full['ic']:+.4f}({s_full['icir']:+.2f}) "
              f"120d_ic={s120['ic']:+.4f}({s120['icir']:+.2f}) 60d_ic={s60['ic']:+.4f}({s60['icir']:+.2f}) "
              f"| adj120={s120['ic']*d:+.4f} adj60={s60['ic']*d:+.4f}")


if __name__ == "__main__":
    main()
