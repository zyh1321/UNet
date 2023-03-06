import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from matplotlib import pyplot as plt
import pylab


seed = torch.random.seed()


def ACDC_dataset(num=0.9, batch_size=4):
    root = glob.glob("./ACDC/database/training/*")
    imgs_path = []
    labels_path = []

    for i in root:
        for j in glob.glob(i + '/*'):
            if 'gt' not in j:
                imgs_path.append(j)
            else:
                labels_path.append(j)
    imgs = []
    labels = []

    for img, label in zip(imgs_path, labels_path):
        img, label = nib.load(img), nib.load(label)
        img, label = img.dataobj, label.dataobj
        for i in range(img.shape[-1]):
            # 去除无病灶和分类数超过四的数据
            if np.max(label[:, :, i]).all() > 0 and np.unique(label[:, :, i]).all() < 5:
                imgs.append(img[:, :, i])
                labels.append(label[:, :, i])

    # 切分训练集，测试集
    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    np.random.shuffle(labels)
    s = int(len(imgs) * num)

    train_imgs = imgs[:s]
    train_labels = labels[:s]

    test_imgs = imgs[s:]
    test_labels = labels[s:]

    non_transforms = transforms.Compose([
        transforms.CenterCrop((128, 128))
    ])

    train_loader = ACDC_data(train_imgs, train_labels, non_transforms)
    test_loader = ACDC_data(test_imgs, test_labels, non_transforms)

    # 数据增强
    change_transforms = transforms.Compose([
        transforms.CenterCrop((128, 128)),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1)

    ])
    enhance_loader = ACDC_data(train_imgs, train_labels, change_transforms)
    train_loader.imgs.extend(enhance_loader.imgs)
    train_loader.labels.extend(enhance_loader.labels)

    train_data = DataLoader(train_loader, batch_size, shuffle=True)
    test_data = DataLoader(test_loader, batch_size, shuffle=True)

    return train_data, test_data


class ACDC_data(Dataset):
    def __init__(self, imgs, labels, transforms):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

        self.idx = {0: 0, 1: 85, 2: 170, 3: 255}

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = np.array(img)

        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        torch.random.manual_seed(seed)
        img_tensor = self.transforms(img_tensor)

        label_np = np.array(label)
        # 将像素值替换成标签
        label = label_np.copy()
        for k, v in self.idx.items():
            label_np[label == v] = k
        # 将字典中未包含的像素值转成 0
        label_np = np.where(label_np > 4, 0, label_np)

        label_tensor = torch.from_numpy(label_np)
        torch.random.manual_seed(seed)
        label_tensor = self.transforms(label_tensor)
        label_tensor = torch.squeeze(label_tensor).type(torch.FloatTensor)

        return torch.unsqueeze(img_tensor, 0), label_tensor

    def __len__(self):
        return len(self.imgs)


# if __name__ == '__main__':
#     train_data, test_data = ACDC_dataset(num=0.9, batch_size=8)
#     image, label = next(iter(train_data))
#     plt.figure(figsize=(25, 25))
#     column = 4
#     for i in range(label.shape[0]):
#         # 打印原始图像
#         plt.subplot(column, 15, i + 1)
#         plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
#         if i == 0:
#             plt.annotate('Image   ', (-1.12, 0.5), xycoords='axes fraction', fontsize=18,
#                          va='center', rotation=0)
#         plt.axis('off')
#         # 打印标签图像
#         plt.subplot(column, 15, i + 16)
#         plt.imshow(label[i].cpu().numpy())
#         if i == 0:
#             plt.annotate('Label   ', (-1.12, 0.5), xycoords='axes fraction', fontsize=18,
#                          va='center', rotation=0)
#         plt.axis('off')
#     pylab.show()
