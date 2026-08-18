"""
miner_1 cycle22 batchA: trend-persistence family (distance-from-high drawdown position).
Idea: assets trading near their multi-month highs exhibit persistent trend continuation;
assets far below highs are in corrective regimes. Revalidated freshly through 2026-08-13
because the previous library (yield_beta_cond_60x20 etc.) that evicted hl_pos is gone;
the only effective factor is now usdcny_beta_60 (a macro-beta), expected low correlation.
Candidates: hl_pos_150, hl_pos_180, hl_pos_220, hl_neg_120 (close/rolling_min-1).
No lookahead: factor uses data up to t, forward returns t..t+h.
"""
import sys, json, base64, zlib, io, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import factor_validation_lib as fvl

END = pd.Timestamp("2026-08-13")
fvl.CURRENT_DATE = END
close, vol, open_, high, low = fvl.load_closes(END)
market = close.pct_change().mean(axis=1, skipna=True)
disp = close.pct_change().std(axis=1, skipna=True)
macro = {"market": market, "disp": disp}


def hl_pos(close, vol, open_, high, low, macro, window=180):
    roll_max = close.rolling(window, min_periods=max(40, window // 3)).max()
    return close / roll_max - 1.0


def hl_neg(close, vol, open_, high, low, macro, window=120):
    roll_min = close.rolling(window, min_periods=max(40, window // 3)).min()
    return close / roll_min - 1.0


CANDIDATES = {
    "hl_pos_150": lambda *a, **k: hl_pos(*a, **k, window=150),
    "hl_pos_180": lambda *a, **k: hl_pos(*a, **k, window=180),
    "hl_pos_220": lambda *a, **k: hl_pos(*a, **k, window=220),
    "hl_neg_120": lambda *a, **k: hl_neg(*a, **k, window=120),
}


def load_panel(path):
    d = json.load(open(path))
    art = d["validation"]["signal_artifact"]
    raw = base64.b64decode(art["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()),
                    index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return p


lib_panels = {}
try:
    lib_panels["usdcny_beta_60"] = load_panel("factors/usdcny_beta_60.json")
except Exception as e:
    print(f"[warn] cannot load usdcny_beta_60: {e}")

results, panels = {}, {}
for name, fn in CANDIDATES.items():
    res = fvl.validate_factor(fn, close, vol, open_, high, low, macro,
                              horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10)
    results[name] = res
    panels[name] = res["panel"]
    fvl.print_result(name, res)

print("\n=== library correlation vs usdcny_beta_60 ===")
for name in CANDIDATES:
    rho = fvl.max_library_corr(panels[name], lib_panels)
    results[name]["rho_vs_usdcny_beta_60"] = round(rho, 4)
    print(f"  {name}: max_abs_rho={rho:.4f}")

print("\n=== regime / recency IC (10d horizon) ===")
fr10 = close.pct_change(10).shift(-10)
for name in CANDIDATES:
    regs = {}
    for label, lo, hi in [("2020-2021", "2020-01-01", "2021-12-31"),
                          ("2022-2023", "2022-01-01", "2023-12-31"),
                          ("2024-2026-08", "2024-01-01", "2026-08-13"),
                          ("recent3m", "2026-05-13", "2026-08-13")]:
        sub = panels[name].loc[lo:hi]
        frs = fr10.loc[lo:hi]
        ics = []
        for dt in sub.index:
            x, y = sub.loc[dt], frs.loc[dt]
            m = x.notna() & y.notna()
            if m.sum() >= 8:
                ics.append(x[m].rank().corr(y[m].rank()))
        if ics:
            s = pd.Series(ics)
            regs[label] = [round(float(s.mean()), 4), round(float(s.mean() / s.std()), 4), len(s)]
        else:
            regs[label] = None
    print(f"  {name}: {regs}")

json.dump({name: {k: v for k, v in results[name].items() if k != "panel"}
           for name in CANDIDATES},
          open("scripts/_miner1_cycle22_batchA_results.json", "w"), indent=1, default=str)
print("\ndone A.")