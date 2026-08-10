import os
print('memory exists:', os.path.exists('memory.txt'))
if os.path.exists('memory.txt'):
    print('size:', os.path.getsize('memory.txt'))
    print('mtime:', os.path.getmtime('memory.txt'))
for f in ['scripts/miner3_20260716_screen_cycle7.py','scripts/miner3_20260716_screen_cycle7b.py']:
    print(f, os.path.exists(f))
print('cwd:', os.getcwd())
print(os.listdir('.'))