"""Cycle6b: screen second batch of diversification / trend-guard candidates (miner_2, 2026-10-30).

Motivation: Screener feedback asked for a trend/momentum guard; the first batch
(mom10d_clean, trend_ma10_30, vol_mom10d) FAILED the daily-IC gate. Test more
interpretable candidates that could complement the reversal family on the
15-asset cross-asset panel:

  mom20_skip5    : ln(c[-5]/c[-25])                (20d momentum, 5d skip)
  mom60_skip20   : ln(c[-20]/c[-80])               (medium-term momentum, skip 20)
  lowvol20       : -std20(log ret)                 (low-vol premium, negated)
  volscaled_rev1 : -ret1 / std20                   (vol-scaled reversal)
  hl_range_20    : mean((h-l)/c, 20)               (range/volatility level)

Gates (15-instrument cross-asset universe):
  |daily mean IC| >= 0.0070 ; |daily ICIR| >= 0.0840 ; date needs >= 8 valid.
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
    ret = c.pct_change()
    if fid == "mom20_skip5":
        x = np.log(c.shift(5) / c.shift(25))
    elif fid == "mom60_skip20":
        x = np.log(c.shift(20) / c.shift(80))
    elif fid == "lowvol20":
        x = -ret.rolling(20).std()
    elif fid == "volscaled_rev1":
        x = -np.log(c / c.shift(1)) / ret.rolling(20).std()
    elif fid == "hl_range_20":
        x = ((h - l) / c).rolling(20).mean()
    else:
        raise ValueError(fid)
    return x.replace([np.inf, -np.inf], np.nan)


def main():
    panels = {}
    for a in ASSETS:
        df = get_stock_daily_data(symbol=a, days=DAYS)
        if df is None or len(df) < 120:
            print(f"SKIP {a}: insufficient data ({None if df is None else len(df)})")
            continue
        df = df.copy()
        df["date"] = df["date"].astype(str)
        df["ret1"] = np.log(df["close"].astype(float).shift(-1) / df["close"].astype(float))
        df["ret5"] = np.log(df["close"].astype(float).shift(-5) / df["close"].astype(float))
        df["ret10"] = np.log(df["close"].astype(float).shift(-10) / df["close"].astype(float))
        panels[a] = df.set_index("date")
    print(f"Loaded {len(panels)}/15 assets")

    all_dates = sorted(set().union(*[set(p.index) for p in panels.values()]))
    dates = pd.Index(all_dates)
    print(f"Common date range: {all_dates[0]} .. {all_dates[-1]} ({len(all_dates)} dates)")

    factors = ["mom20_skip5", "mom60_skip20", "lowvol20", "volscaled_rev1", "hl_range_20"]

    rows = []
    for fid in factors:
        fdf = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
        r1 = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
        r5 = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
        r10 = pd.DataFrame(index=dates, columns=ASSETS, dtype=float)
        for a, p in panels.items():
            f = factor_series(p, fid).reindex(dates)
            fdf[a] = f.values
            r1[a] = p["ret1"].reindex(dates).values
            r5[a] = p["ret5"].reindex(dates).values
            r10[a] = p["ret10"].reindex(dates).values
        fdf = fdf.rank(axis=1)
        nv = fdf.notna().sum(axis=1)

        def ic_panel(retmat):
            out = []
            for i in range(len(dates)):
                out.append(spearman(fdf.iloc[i].values, retmat.iloc[i].values))
            return np.array(out)

        ic1 = ic_panel(r1); ic5 = ic_panel(r5); ic10 = ic_panel(r10)
        full = np.isfinite(ic1)
        recent = np.zeros_like(full); recent[-120:] = full[-120:] if full[-120:].any() else False
        w_full = ic1[full]; w_rec = ic1[recent]
        mf = np.nanmean(w_full); sf = np.nanstd(w_full)
        mr = np.nanmean(w_rec); sr = np.nanstd(w_rec)
        rankdf = fdf
        turn = float(np.nanmean(np.abs(rankdf.diff().values)))
        rows.append({
            "factor": fid, "n_full": int(full.sum()), "n_rec": int(recent.sum()),
            "ic_full": mf, "icir_full": mf / sf if sf > 0 else np.nan,
            "hit_full": float(np.mean(np.sign(w_full) == np.sign(mf))) if len(w_full) else np.nan,
            "ic_rec": mr, "icir_rec": mr / sr if sr > 0 else np.nan,
            "ic5": np.nanmean(ic5[full]), "ic10": np.nanmean(ic10[full]),
            "turnover": turn, "coverage": float(np.mean(nv >= MIN_VALID)),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    print("\n=== SCREEN RESULTS (daily 1d-forward Spearman IC, 15 assets) ===")
    print(out.to_string(index=False))

    print("\n=== GATE CHECK (|IC|>=0.0070, |ICIR|>=0.0840) ===")
    for _, r in out.iterrows():
        gf = abs(r["ic_full"]) >= 0.0070 and abs(r["icir_full"]) >= 0.0840
        gr = abs(r["ic_rec"]) >= 0.0070 and abs(r["icir_rec"]) >= 0.0840
        print(f"{r['factor']:<16} full={'PASS' if gf else 'fail'} "
              f"(ic={r['ic_full']:+.4f}, icir={r['icir_full']:+.4f}, n={int(r['n_full'])}) | "
              f"recent={'PASS' if gr else 'fail'} "
              f"(ic={r['ic_rec']:+.4f}, icir={r['icir_rec']:+.4f}, n={int(r['n_rec'])}) | "
              f"cov={r['coverage']:.2f} turn={r['turnover']:.2f} ic5={r['ic5']:+.4f} ic10={r['ic10']:+.4f}")


if __name__ == "__main__":
    main()
