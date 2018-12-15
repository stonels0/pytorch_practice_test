import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        '''目标：获取所有图片的地址，并且根据训练、验证、测试来划分数据集'''
        self.test = test
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1 : data/test1/11101.jpg
        # train:  data/cat.11123.jpg
        if self.test:
            imgs = sorted(
                imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练、验证集，训练：验证 = 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:

            # 当未定义相对应的图像预处理规则时，进行下面的操作
            # 数据转化操作，测试验证 和 训练的数据转化有所区分， 训练会进行数据增强操作，如随机裁剪、随机翻转、加噪声等操作

            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 测试集和验证集数据预处理
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                ])
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizeCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(), normalize
                ])

        def __getitem__(self, index):
            '''
            返回一张图片的数据
            对于测试集，没有label，返回图片的 id ，如12000.jpg 则返回 12000
            '''
            img_path = self.imgs[index]
            if self.test:
                label = int(self.imgs[index].split('.')[-2].split('/')[-1])
            else:
                label = 1 if 'dog' in img_path.split('/')[-1] else 0

            data = Image.open(img_path)
            data = self.transforms(data)

            return data, label

        def __len__(self):
            return len(imgs)
