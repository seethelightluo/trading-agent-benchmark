
with open('strategy.py') as f:
    lines = f.readlines()
for i in range(185, 300):
    print(f'{i+1}: {lines[i]}', end='')
