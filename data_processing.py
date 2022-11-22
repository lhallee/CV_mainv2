import numpy as np
import os
import torch
import torchvision
from torch.utils import data
from skimage.util import view_as_windows
from tqdm import tqdm
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class ImageSet(data.Dataset):
    def __init__(self, imgs, GTs):
        self.imgs = imgs
        self.GTs = GTs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.imgs[index]
        GT = self.GTs[index]
        return img, GT

def crop_augment(img_path, GT_path, dim, step, num_class):
    img = np.array(Image.open(img_path)) / 255.0
    GT = np.array(Image.open(GT_path))
    a, b = GT.shape
    GT = GT.reshape(a, b, 1)
    imgs = view_as_windows(img, (dim, dim, 3), step=step)
    GTs = view_as_windows(GT, (dim, dim, 1), step=step)
    a, b, c, d, e, f = imgs.shape
    imgs = imgs.reshape(a * b, dim, dim, 3)
    GTs = GTs.reshape(a * b, dim, dim, 1)
    if num_class == 2:
        GTs[GTs < 1] = 0
        GTs[GTs > 0] = 1
    if num_class == 3:
        GTs[GTs == 0] = 1
        GTs = GTs - 1
    delete_list = []
    for i in range(len(imgs)):
        per = (np.count_nonzero(np.array(GTs[i])) / (dim * dim)) * 100
        if (per < 0.01):
            delete_list.append(i)
        else:
            continue
    imgs = np.delete(imgs, delete_list, 0)
    GTs = np.delete(GTs, delete_list, 0)
    #GTs = to_categorical(GTs, num_classes=num_class)
    imgs_90 = np.copy(imgs)
    imgs_vflip = np.copy(imgs)
    imgs_hflip = np.copy(imgs)
    imgs_jitter = torch.tensor(np.transpose(np.copy(imgs), axes=(0, 3, 1, 2)), dtype=torch.float32)
    GTs_90 = np.copy(GTs)
    GTs_vflip = np.copy(GTs)
    GTs_hflip = np.copy(GTs)
    GTs_jitter = np.copy(GTs)
    for i in range(len(imgs)):
        imgs_90[i] = np.rot90(imgs_90[i])
        imgs_vflip[i] = np.flipud(imgs_vflip[i])
        imgs_hflip[i] = np.fliplr(imgs_hflip[i])
        transform = torchvision.transforms.ColorJitter(np.random.uniform(0.0, 0.4),
                                                       np.random.uniform(0.0, 0.4),
                                                       np.random.uniform(0.0, 0.4),
                                                       np.random.uniform(0.0, 0.4))
        imgs_jitter[i] = transform(imgs_jitter[i])
        GTs_90[i] = np.rot90(GTs_90[i])
        GTs_vflip[i] = np.flipud(GTs_vflip[i])
        GTs_hflip[i] = np.fliplr(GTs_hflip[i])
    imgs_jitter = np.transpose(np.array(imgs_jitter), axes=(0, 2, 3, 1))
    final_crops = np.concatenate((imgs, imgs_90, imgs_vflip, imgs_hflip, imgs_jitter), axis=0)
    final_crops_GT = np.concatenate((GTs, GTs_90, GTs_vflip, GTs_hflip, GTs_jitter), axis=0)
    return final_crops, final_crops_GT

def file_to_dataloader(img_path, GT_path,
                       dim=256, num_class=2, train_per=0.7,
                       batch_size=8, num_cpu=os.cpu_count()):
    img_paths = sorted(glob(img_path + '*.png'))
    GT_paths = sorted(glob(GT_path + '*.png'))
    assert len(img_paths) == len(GT_paths), 'Need GT for every Image.'
    crop_imgs = np.concatenate([crop_augment(img_paths[i], GT_paths[i], dim, int(dim/2), num_class)[0] for i in tqdm(range(len(img_paths)))])
    crop_GTs = np.concatenate([crop_augment(img_paths[i], GT_paths[i], dim, int(dim/2), num_class)[1] for i in tqdm(range(len(img_paths)))])
    crop_imgs = torch.tensor(np.transpose(crop_imgs, axes=(0, 3, 1, 2)), dtype=torch.float)
    crop_GTs = torch.tensor(np.transpose(crop_GTs, axes=(0, 3, 1, 2)), dtype=torch.float)
    X_train, X_mem, y_train, y_mem = train_test_split(crop_imgs, crop_GTs, train_size=train_per)
    X_valid, X_test, y_valid, y_test = train_test_split(X_mem, y_mem, test_size=0.33)
    train_data = ImageSet(X_train, y_train)
    valid_data = ImageSet(X_valid, y_valid)
    test_data = ImageSet(X_test, y_test)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    val_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_cpu)
    return train_loader, val_loader, test_loader

