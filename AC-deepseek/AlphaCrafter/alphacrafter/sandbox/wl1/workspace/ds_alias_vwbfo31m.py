import pathlib
txt = pathlib.Path('scripts/miner1_20300802_build_panel.py').read_text()
print(txt[:2600])