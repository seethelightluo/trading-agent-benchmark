
with open('strategy.py') as f:
    lines = f.readlines()
for i in range(470, min(591, len(lines))):
    print(f'{i+1}: {lines[i]}', end='')
