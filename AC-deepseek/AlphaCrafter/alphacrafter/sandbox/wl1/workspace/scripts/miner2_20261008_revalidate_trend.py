"""Cycle6 factor re-validation + trend-guard screen (miner_2, 2026-10-08).

1) Re-validate the 7 ensemble factors (nclv_1d/2d/3d, rev_1d/2d, nbody_1d,
   mom_10d_skip5) on full sample AND last 120 trading days.
2) Screen trend/momentum guard candidates:
     mom10d_clean  : ln(c/c[-10])                       (strategy v2 overlay)
     trend_ma10_30 : ln(ma10/ma30)
     vol_mom10d    : ln(c/c[-10]) / std20(c)
   for possible library addition per Screener feedback.

Gates (15-instrument cross-asset universe):
  |daily mean IC|  >= 0.0070
  |daily ICIR|     >= 0.0840   (mean/std, daily scale)

Spearman IC via rank+pearson (no scipy). A date needs >= 8 valid instruments.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DAYS = 1800
MIN_VALID = 8


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_VALID:
        return np.nan
    ra = pd.Series(a[m]).rank().values
    rb = pd.Series(b[m]).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def factor_series(df, fid):
    o = df["open"].astype(float); h = df["high"].astype(float)
    l = df["low"].astype(float); c = df["close"].astype(float)
    if fid == "nclv_1d":
        x = -(c - l) / (h - l)
    elif fid == "nclv_2d":
        x = -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
    elif fid == "nclv_3d":
        x = -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
    elif fid == "nbody_1d":
        x = -(c - o) / (h - l)
    elif fid == "rev_1d":
        x = -np.log(c / c.shift(1))
    elif fid == "rev_2d":
        x = -np.log(c / c.shift(2))
    elif fid == "mom_10d_skip5":
        x = np.log(c.shift(5) / c.shift(15))
    elif fid == "mom10d_clean":
        x = np.log(c / c.shift(10))
    elif fid == "trend_ma10_30":
        x = np.log(c.rolling(10).mean() / c.rolling(30).mean())
    elif fid == "vol_mom10d":
        x = np.log(c / c.shift(10)) / (c.pct_change().rolling(20).std())
    else:
        raise ValueError(fid)
    return x.replace([np.inf, -np.inf], np.nan)


def main():
    panels = {}
    for a in ASSETS:
        df = get_stock_daily_data(symbol=a, days=DAYS)
        if df is None or len(df) < 60:
            print(f"SKIP {a}: insufficient data ({None if df is None else len(df)})")
            continue
        df = df.copy()
        df["date"] = df["date"].astype(str)
        df["ret1"] = np.log(df["close"].astype(float).shift(-1) /
                            df["close"].astype(float))
        df["ret5"] = np.log(df["close"].astype(float).shift(-5) /
                            df["close"].astype(float))
        panels[a] = df.set_index("date")
    print(f"Loaded {len(panels)}/15 assets")

    all_dates = sorted(set().union(*[set(p.index) for p in panels.values()]))
    dates = pd.Index(all_dates)
    print(f"Common date range: {all_dates[0]} .. {all_dates[-1]} ({len(all_dates)} dates)")

    factors = ["nclv_1d", "nclv_2d", "nclv_3d", "nbody_1d", "rev_1d", "rev_2d",
               "mom_10d_skip5", "mom10d_clean", "trend_ma10_30", "vol_mom10d"]

    rows = []
    for fid in factors:
        # per-asset factor series aligned to common dates
        fdf = pd.DataFrame(index=dates, dtype=float)
        r1df = pd.DataFrame(index=dates, dtype=float)
        r5df = pd.DataFrame(index=dates, dtype=float)
        for a, p in panels.items():
            s = factor_series(p, fid)
            fdf[a] = s.reindex(dates)
            r1df[a] = p["ret1"].reindex(dates)
            r5df[a] = p["ret5"].reindex(dates)
        ic1 = np.full(len(dates), np.nan)
        ic5 = np.full(len(dates), np.nan)
        nv = np.zeros(len(dates), dtype=int)
        for i, d in enumerate(dates):
            x = fdf.loc[d].values.astype(float)
            y1 = r1df.loc[d].values.astype(float)
            y5 = r5df.loc[d].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y1)
            nv[i] = int(m.sum())
            if m.sum() >= MIN_VALID:
                ic1[i] = spearman(x, y1)
                ic5[i] = spearman(x, y5)
        # turnover: mean abs daily rank change (of factor cross-section ranks)
        rankdf = fdf.rank(axis=1)
        turn = float(rankdf.diff().abs().mean(skipna=True))
        # windows
        full = np.isfinite(ic1)
        recent = np.zeros_like(full)
        recent[-120:] = full[-120:] if full[-120:].any() else False
        w_full = ic1[full]; w_rec = ic1[recent]
        mean_full = np.nanmean(w_full) if len(w_full) else np.nan
        std_full = np.nanstd(w_full) if len(w_full) else np.nan
        icir_full = mean_full / std_full if std_full > 0 else np.nan
        hit_full = float(np.mean(np.sign(w_full) == np.sign(mean_full))) if len(w_full) else np.nan
        mean_rec = np.nanmean(w_rec) if len(w_rec) else np.nan
        std_rec = np.nanstd(w_rec) if len(w_rec) else np.nan
        icir_rec = mean_rec / std_rec if std_rec > 0 else np.nan
        hit_rec = float(np.mean(np.sign(w_rec) == np.sign(mean_rec))) if len(w_rec) else np.nan
        cov = float(np.mean(nv >= MIN_VALID))
        mean5 = np.nanmean(ic5[np.isfinite(ic5)]) if np.isfinite(ic5).any() else np.nan
        rows.append({
            "factor": fid, "n_dates_full": int(full.sum()), "n_dates_rec": int(recent.sum()),
            "ic_full": mean_full, "icir_full": icir_full, "hit_full": hit_full,
            "ic_rec": mean_rec, "icir_rec": icir_rec, "hit_rec": hit_rec,
            "ic5_full": mean5, "turnover": turn, "coverage": cov,
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    print("\n=== RE-VALIDATION / SCREEN RESULTS (1d forward IC, daily) ===")
    print(out.to_string(index=False))

    print("\n=== GATE CHECK (|mean IC|>=0.0070, |ICIR|>=0.0840) ===")
    for _, r in out.iterrows():
        gate_full = abs(r["ic_full"]) >= 0.0070 and abs(r["icir_full"]) >= 0.0840
        gate_rec = abs(r["ic_rec"]) >= 0.0070 and abs(r["icir_rec"]) >= 0.0840
        print(f"{r['factor']:<15} full={'PASS' if gate_full else 'fail'} "
              f"(ic={r['ic_full']:+.4f}, icir={r['icir_full']:+.4f}, n={int(r['n_dates_full'])}) | "
              f"recent={'PASS' if gate_rec else 'fail'} "
              f"(ic={r['ic_rec']:+.4f}, icir={r['icir_rec']:+.4f}, n={int(r['n_dates_rec'])}) | "
              f"cov={r['coverage']:.2f} turn={r['turnover']:.2f} ic5={r['ic5_full']:+.4f}")


if __name__ == "__main__":
    main()
