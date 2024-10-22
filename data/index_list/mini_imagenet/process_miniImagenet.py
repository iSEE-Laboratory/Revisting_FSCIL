import os
import shutil

import numpy as np
import random
import os.path as osp

base_classes = ['n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081',
                'n01843383', 'n01855672', 'n01910747', 'n01930112', 'n01981276',
                'n02074367', 'n02089867', 'n02091244', 'n02091831', 'n02099601',
                'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915',
                'n02110063', 'n02110341', 'n02111277', 'n02113712', 'n02114548',
                'n02116738', 'n02120079', 'n02129165', 'n02138441', 'n02165456',
                'n02174001', 'n02219486', 'n02443484', 'n02457408', 'n02606052',
                'n02687172', 'n02747177', 'n02795169', 'n02823428', 'n02871525',
                'n02950826', 'n02966193', 'n02971356', 'n02981792', 'n03017168',
                'n03047690', 'n03062245', 'n03075370', 'n03127925', 'n03146219',
                'n03207743', 'n03220513', 'n03272010', 'n03337140', 'n03347037',
                'n03400231', 'n03417042', 'n03476684', 'n03527444', 'n03535780']
novel_classes = [
    ['n03544143', 'n03584254', 'n03676483', 'n03770439', 'n03773504'],
    ['n03775546', 'n03838899', 'n03854065', 'n03888605', 'n03908618'],
    ['n03924679', 'n03980874', 'n03998194', 'n04067472', 'n04146614'],
    ['n04149813', 'n04243546', 'n04251144', 'n04258138', 'n04275548'],
    ['n04296562', 'n04389033', 'n04418357', 'n04435653', 'n04443257'],
    ['n04509417', 'n04515003', 'n04522168', 'n04596742', 'n04604644'],
    ['n04612504', 'n06794110', 'n07584110', 'n07613480', 'n07697537'],
    ['n07747607', 'n09246464', 'n09256479', 'n13054560', 'n13133613'],
]


def main():
    seed = 2
    np.random.seed(seed)
    root = './'
    N, K = 5, 5
    output_path = f'./new_split_seed{seed}'
    if not osp.exists(output_path):
        os.makedirs(output_path)

    shutil.copyfile(osp.join(root, 'session_1.txt'), osp.join(output_path, 'session_1.txt'))
    # open csv file
    with open(osp.join(root, 'train.csv'), 'r') as f:
        all_lines = f.readlines()
    all_classes_samples = {}
    for i in all_lines:
        file_name = i.split(',')[0].strip()
        class_name = i.split(',')[1].strip()
        if class_name not in all_classes_samples:
            all_classes_samples[class_name] = [file_name]
        else:
            all_classes_samples[class_name].append(file_name)
    print(all_classes_samples.keys())
    # randomly sample N-way K-shot samples
    for session in range(len(novel_classes)):
        with open(osp.join(output_path, f'session_{session + 2}.txt'), 'w') as f:
            cur_novel_class = novel_classes[session]
            for cls in cur_novel_class:
                all_samples_cur_class = all_classes_samples[cls]
                np.random.shuffle(all_samples_cur_class)
                for i in range(N):
                    f.write(f'MINI-ImageNet/train/{cls}/{all_samples_cur_class[i]}\n')


if __name__ == '__main__':
    main()
