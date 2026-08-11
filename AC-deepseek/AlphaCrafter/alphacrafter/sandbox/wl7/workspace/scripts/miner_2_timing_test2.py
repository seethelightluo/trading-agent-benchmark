import sys, time
sys.path.insert(0, "scripts")
t0 = time.time()
from miner_2_lib import load_panel, load_macro, FACTOR_LAST, fwd_returns, rank_ic_series
import numpy as np, pandas as pd
from scipy.stats import spearmanr

panel = load_panel(); macro = load_macro()
print("load", round(time.time()-t0, 2))

import importlib.util
spec = importlib.util.spec_from_file_location("scr", "scripts/miner_2_20260730_screen_novel_family2.py")
# just import functions by re-executing definitions: use exec on the file up to the functions
src = open("scripts/miner_2_20260730_screen_novel_family2.py").read()
src = src.split('if __name__')[0]
ns = {}
exec(src, ns)
full_library_signals = ns["full_library_signals"]
library_corr_all = ns["library_corr_all"]
print("imported")

t0 = time.time()
libs = full_library_signals(panel, macro)
print("full_library_signals", round(time.time()-t0, 2), "s", list(libs.keys()))

t0 = time.time()
sk = panel.apply(lambda s: s.pct_change().rolling(60).skew())
f = sk.loc[:FACTOR_LAST]
mc, per = library_corr_all(f, panel, macro)
print("library_corr_all", round(time.time()-t0, 2), "s ->", mc, per)
