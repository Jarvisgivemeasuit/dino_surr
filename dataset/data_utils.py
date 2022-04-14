import numpy as np
import os
import random
from time import time
from shutil import copytree
from progress.bar import Bar


def run_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)                  
        cost_time = time() - start
        print("func run time: {:.3f}s.".format(cost_time))
    return wrapper


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@run_time
def get_cls_file(path, num_classes):
    path_list = np.array(os.listdir(path))
    cls_index = range(1000)
    ind_cls_index = random.sample(cls_index, k=num_classes)
    ood_cls_index = np.delete(cls_index, ind_cls_index)
    ind_cls_list, ood_cls_list = path_list[ind_cls_index].tolist(), path_list[ood_cls_index].tolist()

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/ind_cls_{num_classes}.txt', 'w') as f:
        for cls in ind_cls_list:
            f.write(cls+'\n')
        f.close()

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/ood_cls_{1000-num_classes}.txt', 'w') as f:
        for cls in ood_cls_list:
            f.write(cls+'\n')
        f.close

@run_time
def make_ind_datasets(mode, data_domain, num_classes, target_path):
    assert data_domain in ['ind', 'ood'] and num_classes in [100, 500] and mode in ['train', 'val']
    make_sure_path_exists(target_path)
    imagenet_path = f'/mnt/share/cs22-hongly/DATACENTER/ImageNet/{mode}'
    bar = Bar(f'Making {mode}-ind datasets:', max=num_classes)

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/{data_domain}_cls_{num_classes}.txt', 'r') as f:
        for idx, line in enumerate(f):
            cls_name = line.strip()
            copytree(os.path.join(imagenet_path, cls_name), os.path.join(target_path, cls_name))
            bar.suffix = f'[{idx+1}] / [{num_classes}]'
            bar.next()
        bar.finish()


@run_time
def make_ood_dataset(num_classes):
    cls_list = []

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/ood_cls_900.txt', 'r') as f:
        for line in f:
            cls_list.append(line.strip())
    cls_list = np.array(cls_list)

    cls_index = range(900)
    ood_train_index = random.sample(cls_index, k=num_classes)
    ood_valid_index = np.delete(cls_index, ood_train_index)
    ood_train_list, ood_valid_list = cls_list[ood_train_index].tolist(), cls_list[ood_valid_index].tolist()

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/ood_train.txt', 'w') as f:
        for cls in ood_train_list:
            f.write(cls+'\n')
        f.close()

    with open(f'/home/et21-lijl/Documents/OOD_detection/dataset/ood_valid.txt', 'w') as f:
        for cls in ood_valid_list:
            f.write(cls+'\n')
        f.close

    
            

if __name__ == '__main__':
    # path = '/mnt/share/cs22-hongly/DATACENTER/ImageNet/train'
    # get_cls_file(path, 100)
    # mode = 'train'
    # tar_path = f'/home/et21-lijl/Datasets/Imagenet100/{mode}'
    # data_domain = 'ind'
    # num_classes = 100
    # make_ind_datasets(mode, data_domain, num_classes, tar_path)

    # mode = 'val'
    # tar_path = f'/home/et21-lijl/Datasets/Imagenet100/{mode}'
    # make_ind_datasets(mode, data_domain, num_classes, tar_path)
    make_ood_dataset(100)