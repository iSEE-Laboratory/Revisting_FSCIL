import os

root_path = '.'
output_path = '.'

all_list = []
for i in range(1, 12):
    i = 'session_' + str(i) + '.txt'
    print(i)
    with open(i, 'r') as f:
        lines = f.readlines()
    for line in lines:
        path = line.strip('\n')
        idx = int(path.split('/')[2].split('.')[0])
        all_list.append((path, idx))

with open(os.path.join(output_path, 'merged.txt'), 'w') as f:
    for i in all_list:
        sstr = i[0] + ' ' + str(i[1]) + '\n'
        # print(sstr)
        f.write(sstr)

