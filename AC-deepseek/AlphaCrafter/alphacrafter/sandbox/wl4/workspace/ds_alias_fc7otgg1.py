lines = open('memory.txt').readlines()
for i in range(90, 98):
    print('='*20, 'LINE', i, '='*20)
    print(lines[i][:4000])
