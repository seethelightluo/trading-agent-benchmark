"""Test guard-ordering fix + crypto co-downtrend de-rank at 2030-07-05."""
import json, sys
sys.path.insert(0, ".")
import strategy as S

cur, tds = S._today_and_calendar()
acc = S.get_account_dict()
assets = list(acc.get("watch_list", []))
frames = S._fetch(assets)
scores, used = S._scores(frames, assets, cur)
scores2 = S._de_rank_value_traps(dict(scores), frames, assets, cur)
regime = S._regime(frames, assets)
below = S._below_ma(frames, assets)


def pipeline(w, n_alt=4, crypto_bear=False):
    w = S._composite_top2_cap(w, assets, scores2)
    w = S._composite_ma_guard(w, frames, assets)
    w = S._ma_guard(w, frames, assets, cur)
    if crypto_bear:
        w = S._crypto_cap_bear(w, assets, frames)  # hypothetical
    for _ in range(n_alt):
        w = S._commod_cap(w, assets)
        w = S._crypto_cap(w, assets)
    return w


def crypto_cap_bear(w, assets, frames):
    """v14: if BOTH BTC & ETH below MA20 with negative 20d mom -> stricter cap."""
    crypto = [a for a in assets if a in S.CRYPTO and a in w]
    if len(crypto) < 2:
        return w
    d = {}
    for a in crypto:
        df = frames.get(a)
        if df is None or len(df) < 25:
            d[a] = False
            continue
        c = float(df["close"].iloc[-1])
        ma20 = float(df["close"].rolling(20).mean().iloc[-1])
        mom20 = float(df["close"].iloc[-1] / df["close"].iloc[-21] - 1) if len(df) >= 21 else float("nan")
        d[a] = (c < ma20) and mom20 < 0
    if not all(d.get(a, False) for a in crypto):
        return w
    cap = 0.09
    per = 0.055
    csum = sum(w[a] for a in crypto)
    if csum <= cap + 1e-12 and all(w[a] <= per + 1e-12 for a in crypto):
        return w
    # scale down to cap and per-name
    scale = min(1.0, cap / csum, min(per / w[a] for a in crypto if w[a] > 0))
    for a in crypto:
        w[a] *= scale
    excess = csum - sum(w[a] for a in crypto)
    room = [a for a in assets if a not in crypto]
    if room:
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


# baseline current code (v9 then v11 once)
w0 = S._weights(scores2, assets, regime)
w0 = S._composite_top2_cap(w0, assets, scores2)
w0 = S._composite_ma_guard(w0, frames, assets)
w0 = S._ma_guard(w0, frames, assets, cur)
w0 = S._crypto_cap(w0, assets)
w0 = S._commod_cap(w0, assets)

# fix A: alternating 4x
wA = pipeline(S._weights(scores2, assets, regime))

# fix B: alternating 4x + crypto bear cap
wB = pipeline(S._weights(scores2, assets, regime), crypto_bear=True)
# emulate bear cap inside alternating loop
wB = S._weights(scores2, assets, regime)
wB = S._composite_top2_cap(wB, assets, scores2)
wB = S._composite_ma_guard(wB, frames, assets)
wB = S._ma_guard(wB, frames, assets, cur)
for _ in range(4):
    wB = S._commod_cap(wB, assets)
    wB = crypto_cap_bear(wB, assets, frames)

for name, w in [("baseline", w0), ("alt4x", wA), ("alt4x+bearcap", wB)]:
    cs = w.get("BTC", 0) + w.get("ETH", 0)
    com = w.get("WTI", 0) + w.get("COPPER", 0)
    print(f"\n=== {name}: crypto={cs*100:.2f}% commod={com*100:.2f}% sum={sum(w.values()):.6f}")
    for a in sorted(w, key=lambda x: -w[x]):
        print(f"  {a:8s} {w[a]*100:6.2f}%")
