"""miner2 2031-06-06: screen NEW candidate factor ideas on fresh panel (through 2031-06-05)."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')

P = pd.read_pickle('scripts/panel_cache_20310606.pkl')
close, high, low, open_, ret, vol = P['close'], P['high'], P['low'], P['open'], P['ret'], P['vol']
END = str(close.index.max().date())

def fwd_ret(close, h):
    return close.shift(-h) / close - 1.0

def fast_rank_ic(signal, fwd, min_n=8):
    fwd = fwd.reindex(signal.index)
    sig_rank = signal.rank(axis=1)
    fwd_rank = fwd.rank(axis=1)
    mask = signal.notna() & fwd.notna()
    n = mask.sum(axis=1)
    valid = n >= min_n
    sx = sig_rank.where(mask); sy = fwd_rank.where(mask)
    mx = sx.sum(axis=1) / n.replace(0, np.nan)
    my = sy.sum(axis=1) / n.replace(0, np.nan)
    xc = sx.sub(mx, axis=0).where(mask); yc = sy.sub(my, axis=0).where(mask)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1).astype(float) * (yc ** 2).sum(axis=1).astype(float))
    den = den.where(den != 0)
    ic = num / den
    ic = ic[valid].dropna()
    return ic.values, ic.index

def eval_factor(signal, close, horizons=(1, 5, 10), min_n=8, start=None, end=None):
    if start is not None: signal = signal[signal.index >= start]
    if end is not None: signal = signal[signal.index <= end]
    out = {}
    for h in horizons:
        fwd = fwd_ret(close, h)
        ics, dates = fast_rank_ic(signal, fwd, min_n=min_n)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0); continue
        ic = float(np.mean(ics)); sd = float(np.std(ics, ddof=1))
        icir = ic / sd if sd > 0 else np.nan
        hit = float(np.mean(ics > 0))
        out[h] = dict(ic=ic, icir=icir, hit=hit, n=len(ics))
    out['coverage'] = float(signal.notna().mean().mean()) if signal.shape[0] else 0.0
    rp = signal.rank(axis=1, pct=True)
    out['turnover_1d_rank'] = float(rp.diff().abs().mean().mean()) if rp.shape[0] > 1 else 0.0
    out['n_dates'] = int(signal.shape[0])
    return out

def summarize(res, label=''):
    h1, h5, h10 = res.get(1, {}), res.get(5, {}), res.get(10, {})
    print(f"[{label}] IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} "
          f"hit1={h1.get('hit', float('nan')):.3f} n1={h1.get('n', 0)} | "
          f"IC5={h5.get('ic', float('nan')):.4f} ICIR5={h5.get('icir', float('nan')):.3f} | "
          f"IC10={h10.get('ic', float('nan')):.4f} ICIR10={h10.get('icir', float('nan')):.3f} | "
          f"cov={res.get('coverage', float('nan')):.3f} turn1d={res.get('turnover_1d_rank', float('nan')):.3f} "
          f"dates={res.get('n_dates', 0)}")

logret = np.log(close / close.shift(1))
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
ma50 = close.rolling(50).mean()
ma100 = close.rolling(100).mean()
xmed5 = (close / close.shift(5) - 1).median(axis=1)
xmed10 = (close / close.shift(10) - 1).median(axis=1)

cands = {}
cands['rev1_volcond'] = -logret * (vol20 / vol60).replace(0, np.nan)
cands['rev1_hivol'] = -logret * (vol20 > vol60).astype(float)
cands['rev5_volcond'] = -(np.log(close) - np.log(close.shift(5))) * (vol20 / vol60).replace(0, np.nan)
cands['rev5_hivol'] = -(np.log(close) - np.log(close.shift(5))) * (vol20 > vol60).astype(float)
cands['dd_20d'] = -(close / close.rolling(20).max() - 1.0)
cands['dd_60d'] = -(close / close.rolling(60).max() - 1.0)
cands['rev_120d'] = -(close / close.shift(120) - 1.0)
cands['mom20_trend'] = (close / close.shift(20) - 1.0) * np.sign(close - ma50).fillna(0.0)
cands['mom60_trend'] = (close / close.shift(60) - 1.0) * np.sign(close - ma100).fillna(0.0)
cands['volz_neg'] = -(vol / vol.rolling(20).mean() - 1.0)
cands['volz_pos'] = (vol / vol.rolling(20).mean() - 1.0)
cands['relstr_10d'] = (close / close.shift(10) - 1.0) - xmed10
cands['relstr_5d'] = (close / close.shift(5) - 1.0) - xmed5
cands['lower_wick'] = (np.minimum(open_, close) - low) / (high - low).replace(0, np.nan)
cands['upper_wick_neg'] = -(high - np.maximum(open_, close)) / (high - low).replace(0, np.nan)
cands['range_20_neg'] = -((high.rolling(20).max() - low.rolling(20).min()) / close)
cands['body_vol_norm'] = ((close - open_) / close) / vol20.replace(0, np.nan)

FULL_START, FULL_END = '2021-01-01', END
RECENT2_START = '2030-06-01'

print("=" * 150)
print(f"NEW FACTOR SCREEN | FULL {FULL_START}..{FULL_END} | RECENT2 {RECENT2_START}..{END}")
print("=" * 150)
out = {}
for name, sig in cands.items():
    full = eval_factor(sig, close, start=FULL_START, end=FULL_END)
    recent2 = eval_factor(sig, close, start=RECENT2_START, end=END)
    out[name] = dict(full=full, recent2=recent2)
    summarize(full, f"{name} FULL")
    summarize(recent2, f"{name} R2")
    print("-" * 150)

with open('scripts/miner2_screen_20310606.json', 'w') as f:
    json.dump(out, f, indent=1, default=float)
print("saved scripts/miner2_screen_20310606.json")
