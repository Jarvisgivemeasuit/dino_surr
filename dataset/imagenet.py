import os
import cv2
import random
import albumentations as A
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image


class Imagenet(Dataset):
    '''
    Generating a torch dataset using Imagenet for training or validation.
    Args:
        mode: Specifies the dataset to train or test.
        data_domain: Determines this dataset as IND or OOD.
    '''
    def __init__(self, mode, num_classes=None, args=None) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.args = args
        self.num_classes = num_classes
        self.imagenet_path = f'/mnt/share/cs22-hongly/DATACENTER/ImageNet/{mode}'
        self.classes, self.img_list = {}, []

        with open(f'/home/et21-lijl/Documents/dino/dataset/ind_cls_{self.num_classes}.txt', 'r') as f:
            for idx, line in enumerate(f):
                cls_name = line.strip()
                self.classes[cls_name] = [idx, 0]

                cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                self.img_list = self.img_list + cls_img_list

        with open(f'/home/et21-lijl/Documents/dino/dataset/ood_{self.mode}.txt', 'r') as f:
            for idx, line in enumerate(f):
                cls_name = line.strip()
                self.classes[cls_name] = [idx, 1]

                cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                self.img_list = self.img_list + cls_img_list


    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        cls_name = img_name.split('/')[0]
        cls_label, domain_label = self.classes[cls_name]

        img = cv2.imread(os.path.join(self.imagenet_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.train_transformations(img) if self.mode == 'train' else \
                    self.valid_transformations(img)
        img = img.transpose(2, 0, 1)
        return img, cls_label, domain_label

    def __len__(self):
        return len(self.img_list)

    def train_transformations(self, image):
        compose = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Resize(500, 500, p=1),
            A.RandomSizedCrop((300, 500), 224, 224, p=1),
            A.Normalize()
        ])
        return compose(image=image)['image']

    def valid_transformations(self, image):
        norm = A.Compose([
            A.Resize(224, 224, p=1),
            A.Normalize()
        ])
        return norm(image=image)['image']
