# Patch round9 script: wrap each candidate in try/except so one bad panel doesn't abort the batch
src = open('scripts/miner_3_20260730_screen_round9_novel.py').read()
old = """for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    panels[fid] = panel
    m = validate_factor(fid, panel, prices)"""
new = """for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    panels[fid] = panel
    try:
        m = validate_factor(fid, panel, prices)
    except Exception as e:
        print("%s: VALIDATION ERROR %s; panel %s; treating as FAIL" % (fid, e, panel.shape))
        results[fid] = dict(ok=False, metrics={'ic': float('nan'), 'icir': float('nan'), 'error': str(e)})
        continue"""
assert old in src, "pattern not found"
src = src.replace(old, new)
open('scripts/miner_3_20260730_screen_round9_novel.py', 'w').write(src)
print("patched OK")
