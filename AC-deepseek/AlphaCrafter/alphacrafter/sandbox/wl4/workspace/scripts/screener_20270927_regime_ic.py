"""Screener 2027-09-27: market regime assessment + recent rank IC for active library
and evicted candidates (visible through 2027-09-24, the previous completed trading day)."""
import csv, math
from collections import defaultdict
from datetime import datetime, date

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
OBS = ["DXY", "EURUSD", "USDCNY", "USDJPY", "VIX"]
CUTOFF = date(2027, 9, 24)   # previous completed trading day before decision 2027-09-27
START = date(2026, 7, 16)    # online start

def load_close(sym, idx=False):
    px, vol = {}, {}
    root = "../persistent/index_data" if idx else "../persistent/stock_data"
    with open(f"{root}/{sym}.csv") as f:
        for row in csv.DictReader(f):
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            if d > CUTOFF:
                continue
            try:
                px[d] = float(row["close"])
            except (TypeError, ValueError):
                continue
            try:
                v = float(row["volume"])
                vol[d] = v
            except (TypeError, ValueError):
                vol[d] = 0.0
    return px, vol

closes, vols = {}, {}
for a in ASSETS:
    closes[a], vols[a] = load_close(a)
obs = {}
for m in OBS:
    obs[m], _ = load_close(m, idx=True)

def daily_ret(px):
    out = {}
    ds = sorted(px)
    for a, b in zip(ds, ds[1:]):
        if px[a] and px[b]:
            out[b] = px[b] / px[a] - 1.0
    return out

rets = {a: daily_ret(closes[a]) for a in ASSETS}
obs_ret = {m: daily_ret(obs[m]) for m in OBS}

print("=" * 78)
print("MARKET REGIME ASSESSMENT  (through 2027-09-24)")
print("=" * 78)
all_dates = sorted(set().union(*[set(r) for r in rets.values()]))
recent = [d for d in all_dates if d >= date(2027, 3, 1)]

# trend & vol per asset
print("\n%-10s %8s %8s %8s %8s %9s %8s %8s" % (
    "asset", "ret20d", "ret60d", "ret120d", "vs60ma", "annvol20", "dd120", "10d-fwd"))
fwd10 = {}
for a in ASSETS:
    ds = sorted(closes[a])
    pos = {d: i for i, d in enumerate(ds)}
    fr = {}
    for d in ds:
        j = pos[d] + 10
        if j < len(ds):
            fr[d] = closes[a][ds[j]] / closes[a][d] - 1.0
    fwd10[a] = fr

def pct(s, d0, k):
    ds = sorted(s)
    if d0 not in pos0: return float('nan')
    i = pos0[d0]
    j = i - k
    if j < 0: return float('nan')
    return s[d0] / s[ds[j]] - 1.0

pos0 = {d: i for i, d in enumerate(sorted(closes["SPX"]))}
for a in ASSETS:
    ds = sorted(closes[a])
    last = ds[-1]
    def ret_k(k):
        i = pos0[last]
        j = i - k
        if j < 0 or j >= len(ds): return float('nan')
        return closes[a][last] / closes[a][ds[j]] - 1.0
    r20, r60, r120 = ret_k(20), ret_k(60), ret_k(120)
    # 60d MA
    i = pos0[last]
    win = ds[max(0, i-59): i+1]
    ma60 = sum(closes[a][d] for d in win) / len(win)
    vsma = closes[a][last] / ma60 - 1.0
    # ann vol 20d
    rs = [rets[a][d] for d in ds[-20:] if d in rets[a]]
    ann = (sum(rs)/len(rs) / 1.0)  # placeholder
    sd = (sum((x - sum(rs)/len(rs))**2 for x in rs) / len(rs)) ** 0.5 if len(rs) > 1 else 0
    annvol = sd * math.sqrt(252)
    # max dd 120d
    win2 = ds[max(0, i-119): i+1]
    peak = -1e18; mdd = 0.0
    for d in win2:
        peak = max(peak, closes[a][d])
        mdd = min(mdd, closes[a][d] / peak - 1.0)
    fv = fwd10[a].get(last, float('nan'))
    print("%-10s %8.1f%% %8.1f%% %8.1f%% %8.1f%% %9.1f%% %8.1f%% %8.1f%%" % (
        a, r20*100, r60*100, r120*100, vsma*100, annvol*100, mdd*100, fv*100))

# macro obs
print("\nMacro observation-only:")
for m in OBS:
    ds = sorted(obs[m])
    last = ds[-1]
    def ret_k(k):
        i = pos0[last]
        j = i - k
        if j < 0 or j >= len(ds): return float('nan')
        return obs[m][last] / obs[m][ds[j]] - 1.0
    print("  %-8s last=%-10s ret20d=%+.2f%%  ret60d=%+.2f%%" % (
        m, last, ret_k(20)*100, ret_k(60)*100))

# cross-sectional dispersion & pairwise correlation (20d)
print("\nCross-sectional stats (20d window ending 2027-09-24):")
ds20 = recent[-20:] if len(recent) >= 20 else recent
xs_ret = []
for d in ds20:
    vals = [rets[a][d] for a in ASSETS if d in rets[a]]
    if len(vals) >= 8:
        xs_ret.append((d, vals))
disp = []
for d, vals in xs_ret:
    m = sum(vals)/len(vals)
    disp.append(math.sqrt(sum((v-m)**2 for v in vals)/len(vals)))
print("  mean cross-sectional std of daily rets (20d): %.4f%%" % (sum(disp)/len(disp)*100 if disp else float('nan')))

# avg pairwise corr of daily returns over last 60 trading days
import itertools
pc = []
for d in ds20:
    vec = {}
    for a in ASSETS:
        if d in rets[a]:
            vec[a] = rets[a][d]
    if len(vec) >= 8:
        names = list(vec.keys())
        for x, y in itertools.combinations(names, 2):
            pc.append((x, y, vec[x], vec[y]))
# aggregate corr per pair over window
pair_map = defaultdict(list)
for x, y, vx, vy in pc:
    pair_map[(x, y)].append((vx, vy))
corrs = []
for (x, y), pairs in pair_map.items():
    if len(pairs) < 10: continue
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
    if den > 0:
        corrs.append(num/den)
print("  avg pairwise corr (20d): %.3f  (n_pairs=%d)" % (sum(corrs)/len(corrs) if corrs else float('nan'), len(corrs)))

# ================= FACTOR RECENT IC =================
print("\n" + "=" * 78)
print("RECENT RANK IC (h=10) — online period 2026-07-16..2027-09-24")
print("=" * 78)

def rolling_beta(asset_r, ref_r, win=60, min_obs=40, down_only=False):
    out = {}
    ds = sorted(set(asset_r) & set(ref_r))
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        xs, ys = [], []
        for dd in window:
            x = ref_r[dd]
            if down_only and x >= 0:
                continue
            xs.append(x); ys.append(asset_r[dd])
        if len(xs) < min_obs: continue
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        den = sum((x-mx)**2 for x in xs)
        if den == 0: continue
        out[d] = num/den
    return out

def corr(xs, ys):
    n = len(xs)
    if n < 3: return float("nan")
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num/den if den else float("nan")

mkt_ret = {}
for d in all_dates:
    vals = [rets[a][d] for a in ASSETS if d in rets[a]]
    if len(vals) >= 8:
        mkt_ret[d] = sum(vals)/len(vals)
cn10y_ret = rets["CN10Y"]
eur_ret = obs_ret["EURUSD"]

def f_vol_price_corr(a, win=20, min_obs=10):
    out = {}
    px, v = closes[a], vols[a]
    ds = sorted(px)
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i-win+1): i+1]
        rts, vv = [], []
        for dd in window:
            if dd in rets[a] and v.get(dd, 0) and v[dd] > 0:
                rts.append(rets[a][dd]); vv.append(v[dd])
        if len(rts) < min_obs: continue
        out[d] = corr(rts, vv)
    return out

def f_vol_ratio(a, s=20, l=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(l, len(ds)):
        d = ds[i]
        s20 = [rets[a].get(x, float('nan')) for x in ds[i-s:i]]
        s60 = [rets[a].get(x, float('nan')) for x in ds[i-l:i]]
        s20 = [x for x in s20 if x == x]; s60 = [x for x in s60 if x == x]
        if len(s60) > 30 and len(s20) > 10:
            sd20 = (sum((x - sum(s20)/len(s20))**2 for x in s20)/len(s20))**0.5
            sd60 = (sum((x - sum(s60)/len(s60))**2 for x in s60)/len(s60))**0.5
            if sd60 > 0:
                out[d] = sd20/sd60
    return out

def f_vol_z(a, win=20):
    out = {}
    ds = sorted(closes[a])
    for i in range(win, len(ds)):
        d = ds[i]
        vs = [vols[a].get(x, 0) for x in ds[i-win:i]]
        if all(v > 0 for v in vs):
            m = sum(vs)/len(vs)
            sd = (sum((v-m)**2 for v in vs)/len(vs))**0.5
            if sd > 0 and vols[a].get(d, 0) > 0:
                out[d] = (vols[a][d] - m)/sd
    return out

sig = {}
for a in ASSETS:
    sig.setdefault("vol_price_corr_20", {})[a] = f_vol_price_corr(a)
    sig.setdefault("dn_mkt_beta_60d", {})[a] = rolling_beta(rets[a], mkt_ret, 60, 40, down_only=True)
    sig.setdefault("rate_beta_cn10y_60d", {})[a] = rolling_beta(rets[a], cn10y_ret, 60, 40)
    sig.setdefault("eurusd_beta_60d", {})[a] = rolling_beta(rets[a], eur_ret, 60, 40)
    sig.setdefault("vol_ratio_20_60", {})[a] = f_vol_ratio(a)
    sig.setdefault("volume_z_20", {})[a] = f_vol_z(a)

def forward_ret(px, h=10):
    ds = sorted(px)
    out = {}
    for i, d in enumerate(ds):
        j = i + h
        if j < len(ds):
            out[d] = px[ds[j]] / px[d] - 1.0
    return out

fwd = {a: forward_ret(closes[a], 10) for a in ASSETS}

def ranks(v):
    idx = sorted(range(len(v)), key=lambda k: v[k])
    r = [0]*len(v)
    for rank, pos in enumerate(idx):
        r[pos] = rank
    return r

def rank_ic(fv, d):
    xs, ys = [], []
    for a in ASSETS:
        if d in fv.get(a, {}) and d in fwd.get(a, {}):
            x, y = fv[a][d], fwd[a][d]
            if x == x and y == y:
                xs.append(x); ys.append(y)
    if len(xs) < 8: return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return num/den if den else None

for fname in ["vol_price_corr_20", "dn_mkt_beta_60d", "rate_beta_cn10y_60d",
              "eurusd_beta_60d", "vol_ratio_20_60", "volume_z_20"]:
    ic_dates = []
    for d in all_dates:
        if d < START or d > CUTOFF: continue
        ic = rank_ic(sig[fname], d)
        if ic is not None:
            ic_dates.append((d, ic))
    if not ic_dates:
        print(f"{fname}: no IC dates"); continue
    ic_dates.sort()
    n = len(ic_dates)
    print(f"\n{fname}:")
    for lab, k in [("all", n), ("last250", min(250, n)), ("last120", min(120, n)), ("last60", min(60, n))]:
        sub = ic_dates[-k:]
        m = sum(x[1] for x in sub)/len(sub)
        sd = (sum((x[1]-m)**2 for x in sub)/len(sub))**0.5 if len(sub) > 1 else 0
        icir = m/sd if sd else 0
        hit = sum(1 for x in sub if x[1] > 0)/len(sub)
        print(f"  [{lab:7s}] n={len(sub):4d} meanIC={m:+.4f} ICIR={icir:+.3f} hit={hit:.2f}")
    print(f"  last date: {ic_dates[-1][0]}  last IC={ic_dates[-1][1]:+.4f}")

print("\nDONE")
