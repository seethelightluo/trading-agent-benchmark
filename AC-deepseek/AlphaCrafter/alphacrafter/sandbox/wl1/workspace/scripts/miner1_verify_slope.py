import sys, numpy as np, pandas as pd

src = open("scripts/miner1_20260716_screen_cycle_riskvol_v2.py").read()
head = src.split("# ---------------------------------------------------------------- candidates")[0]
ns = {"np": np, "pd": pd}
exec(head, ns)
fn = ns["slope_tstat_vec"](20)

rng = np.random.default_rng(0)
n = 300
close = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
close[150:160] = np.nan
df = pd.DataFrame({"close": close}, index=pd.date_range("2025-01-01", periods=n))
got = fn(df).values

ref = np.full(n, np.nan)
for i in range(19, n):
    w = df["close"].iloc[i - 19:i + 1].dropna().values
    if len(w) < 20:
        continue
    x = np.arange(len(w), dtype=float)
    out = np.polyfit(x, w, 1, full=True)
    b = out[0][0]
    sse = float(out[1][0]) if len(out[1]) else 0.0
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()
    se_b = np.sqrt(sse / (len(w) - 2) / sxx) if len(w) > 2 else np.nan
    ref[i] = b / se_b if se_b and se_b > 0 else np.nan

m = np.isfinite(got) & np.isfinite(ref)
print("max abs diff vs brute force:", np.abs(got[m] - ref[m]).max())
print("matched finite:", int(m.sum()), "/", n)
