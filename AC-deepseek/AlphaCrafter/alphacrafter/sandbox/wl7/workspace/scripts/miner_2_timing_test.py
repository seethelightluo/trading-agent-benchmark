import sys, time
sys.path.insert(0, "scripts")
t0 = time.time()
from miner_2_lib import load_panel, load_macro
panel = load_panel(); macro = load_macro()
print("load_panel+macro", round(time.time()-t0, 2), "s")
t0 = time.time()
rets = panel.pct_change()
sk = panel.apply(lambda s: s.pct_change().rolling(60).skew())
print("rolling skew 60", round(time.time()-t0, 2), "s")
t0 = time.time()
mkt = rets.mean(axis=1)
beta = rets.rolling(60).cov(mkt) / mkt.rolling(60).var()
print("rolling cov beta", round(time.time()-t0, 2), "s")
t0 = time.time()
v = panel.iloc[:, 0]
am = (v.pct_change().abs() / 1e6).rolling(20).mean()
am2 = am / am.rolling(252).median()
print("amihud small", round(time.time()-t0, 2), "s")
