"""miner_2 2034-12-25: probe data availability + timing for revalidation/screen."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, full_eval, library_signals

t0 = time.time()
panels = load_panels(days=4000)
t1 = time.time()
print(f"load_panels(4000): {t1-t0:.1f}s | symbols: {list(panels.keys())}", flush=True)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"data through: {closes.index.max()} | n_dates: {len(closes)} | n_assets: {closes.shape[1]}", flush=True)
print(f"per-asset rows:\n{closes.count().to_string()}", flush=True)

vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
lib = library_signals(panels, closes, rets, vix)
print("library signals:", list(lib.keys()), flush=True)

# time one full_eval
sig = (closes / closes.shift(20) - 1.0) / rets.rolling(20).std()
t2 = time.time()
m, _ = full_eval(sig, closes, (1, 2, 3, 5, 10, 20), 8, 1, library=lib, admission_horizon=10)
t3 = time.time()
print(f"full_eval one factor: {t3-t2:.1f}s | ic={m['ic']} icir={m['icir']} n={m['n_ic_dates']}", flush=True)
print("PROBE DONE", flush=True)
