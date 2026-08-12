import pathlib, json
# check last memory lines about miner feedback / factor refresh
lines = pathlib.Path('memory.txt').read_text().splitlines()
for l in lines[-4:]:
    print(l[:1500])
    print('======')
