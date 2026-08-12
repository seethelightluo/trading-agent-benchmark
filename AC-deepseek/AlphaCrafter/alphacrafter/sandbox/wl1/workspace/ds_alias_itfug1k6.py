import pathlib
txt = pathlib.Path('scripts/miner1_common.py').read_text()
print(txt[2500:5500])